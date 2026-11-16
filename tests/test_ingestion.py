import pytest

from ma_engine.config import ChunkingConfig
from ma_engine.ingestion.chunker import chunk_document
from ma_engine.ingestion.cleaner import clean_text
from ma_engine.ingestion.loader import decode_bytes, load_bytes, load_text, make_doc_id
from ma_engine.ingestion.preprocess import preprocess
from ma_engine.ingestion.sections import find_headings, section_at


def test_cleaner_removes_noise_and_unwraps():
    raw = (
        "CONFIDENTIAL\n\nARTICLE I\nDEFINITIONS\n\nSection 1.1 Terms. The acquisi-\ntion of the\nCompany shall\n"
        "occur on the go-to-\nmarket date.\n\nPage 1\n_______\n\nCONFIDENTIAL\n\nSection 1.2 More. Text “quoted”.\n"
        "\nCONFIDENTIAL\n\n- 3 -\n\nCONFIDENTIAL\n"
    )
    out = clean_text(raw)
    assert "CONFIDENTIAL" not in out
    assert "Page 1" not in out and "___" not in out and "- 3 -" not in out
    assert "acquisition of the Company shall occur on the go-to-market date." in out
    assert '"quoted"' in out
    assert "ARTICLE I\nDEFINITIONS" in out


def test_cleaner_strips_sgml_and_html():
    raw = "<SEC-HEADER>junk</SEC-HEADER><DOCUMENT><TYPE>EX-2.1<TEXT><p>Merger&nbsp;Agreement</p><br>Body</TEXT>"
    out = clean_text(raw)
    assert "junk" not in out and "<" not in out
    assert "Merger" in out and "Body" in out


def test_cleaner_idempotent(raw_sample):
    once = clean_text(raw_sample.text)
    assert clean_text(once) == once


def test_decode_fallback():
    assert decode_bytes("café".encode("cp1252")) == "café"
    assert decode_bytes("﻿hello".encode("utf-8")) == "hello"


def test_loader_ids_and_errors():
    assert make_doc_id("My Deal", "abc") == make_doc_id("My Deal", "abc")
    assert make_doc_id("My Deal", "abc") != make_doc_id("My Deal", "abd")
    assert load_bytes(b"some text", "deal.txt").title == "deal"
    with pytest.raises(ValueError):
        load_text("   ", "empty")


def test_chunker_sizes_and_overlap():
    words = [f"w{i}." if i % 20 == 19 else f"w{i}" for i in range(2300)]
    text = " ".join(words)
    cfg = ChunkingConfig(chunk_words=500, overlap_words=100)
    chunks = chunk_document("d", text, cfg)
    assert chunks[0].word_start == 0 and chunks[-1].word_end == 2300
    for a, b in zip(chunks, chunks[1:]):
        overlap = a.word_end - b.word_start
        assert 60 <= overlap <= 160, overlap  # ~100 words, snapped to sentence ends
    for c in chunks:
        assert c.word_count <= 500 + 500 // 12 + 1
        assert text[text.index(c.text[:30]):].startswith(c.text[:30])


def test_chunker_cuts_at_paragraph_breaks():
    paras = ["\n\n".join(" ".join(f"p{j}w{i}" for i in range(90)) + "." for j in range(12))]
    chunks = chunk_document("d", paras[0], ChunkingConfig(chunk_words=500, overlap_words=100))
    for c in chunks[:-1]:
        assert c.text.endswith(".")


def test_chunker_small_doc_and_validation():
    assert len(chunk_document("d", "just a few words here", ChunkingConfig())) == 1
    assert chunk_document("d", "", ChunkingConfig()) == []
    with pytest.raises(ValueError):
        chunk_document("d", "x", ChunkingConfig(chunk_words=100, overlap_words=100))


def test_sections_breadcrumbs():
    text = "ARTICLE VIII\n\nTERMINATION\n\nSection 8.3 Parent Termination Fee. If terminated...\nmore"
    hs = find_headings(text)
    assert [h.label for h in hs] == ["ARTICLE VIII - Termination", "Section 8.3 Parent Termination Fee"]
    assert section_at(hs, len(text) - 1) == "ARTICLE VIII - Termination > Section 8.3 Parent Termination Fee"


def test_preprocess_sample(raw_sample):
    doc, chunks = preprocess(raw_sample)
    assert doc.word_count > 2500
    assert len(chunks) >= 6
    assert any("Section 8.3" in s for s in doc.sections)
    assert all(c.doc_id == doc.doc_id for c in chunks)
    assert chunks[1].section.startswith("ARTICLE I")


def test_reassembles_split_document(tmp_path):
    from ma_engine.ingestion.loader import load_folder, load_uploads
    from ma_engine.ingestion.reassembler import find_overlap, piece_key

    original = " ".join(f"Clause {i} of the agreement says something specific." for i in range(300))
    pieces, step, overlap = [], 1000, 200
    for i, start in enumerate(range(0, len(original), step - overlap), start=1):
        pieces.append(original[start:start + step])
        if start + step >= len(original):
            break
    for i, p in enumerate(pieces, start=1):
        (tmp_path / f"contract_0.txt_{i}.txt").write_text(p)
    (tmp_path / "other_deal.txt").write_text("A standalone letter of intent. " * 20)

    docs = load_folder(tmp_path)
    assert len(docs) == 2
    stitched = next(d for d in docs if d.title == "contract_0")
    assert stitched.text == original
    assert "pieces" in stitched.source_path
    assert piece_key(tmp_path / "contract_0.txt_12.txt") == ("contract_0", 12)
    assert piece_key(tmp_path / "deal_part_3.txt") == ("deal", 3)
    assert piece_key(tmp_path / "plain.txt") is None
    assert find_overlap("abcdef" * 10 + "XYZ-tail-overlap-text-here", "XYZ-tail-overlap-text-here and more") == 26

    ups = load_uploads([(f"contract_0.txt_{i}.txt", p.encode()) for i, p in enumerate(pieces, start=1)])
    assert len(ups) == 1 and ups[0].text == original


def test_toc_lines_are_not_headings():
    text = "1.1. Certain Definitions 1\n\n1.2. The Merger 10\n\nArticle VIII GENERAL PROVISIONS 74 iii\n\n1.2. The Merger. At the Effective Time..."
    labels = [h.label for h in find_headings(text)]
    assert labels == ["Section 1.2 The Merger"]


def test_zip_dataset_upload_like_maud():
    import io
    import zipfile

    from ma_engine.ingestion.archives import unpack
    from ma_engine.ingestion.loader import load_uploads

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data/contracts/contract_10.txt", "AGREEMENT AND PLAN OF MERGER among ALPHA CORP., BETA MERGER SUB, "
                   "INC. and GAMMA INC. Dated as of May 1, 2021. " + "Body text. " * 50)
        z.writestr("data/contracts/contract_2.txt", "Another agreement. " * 50)
        z.writestr("data/MAUD_train.csv", "id,question\n1,x")
        z.writestr("__MACOSX/data/._contract_2.txt", "junk")
        z.writestr("data/contracts/empty.txt", "   ")
    result = unpack([("data.zip", buf.getvalue()), ("notes.pdf", b"%PDF"), ("bad.zip", b"not a zip")])
    docs = load_uploads(result.files)
    assert [d.title for d in docs] == ["contract_2", "contract_10"]  # natural order, blank file dropped
    assert any("MAUD_train.csv" in s for s in result.skipped) and "notes.pdf" in result.skipped
    assert any("bad.zip" in s for s in result.skipped)


def test_cover_page_party_labels():
    from ma_engine.ingestion.metadata import describe

    cases = {
        "AGREEMENT AND PLAN OF MERGER dated as of December 12, 2020 among ASTRAZENECA PLC, DELTA OMEGA SUB HOLDINGS "
        "INC., DELTA OMEGA SUB HOLDINGS LLC 2 and ALEXION PHARMACEUTICALS, INC. TABLE OF CONTENTS":
            "Astrazeneca PLC / Alexion Pharmaceuticals, Inc.",
        "AGREEMENT AND PLAN OF MERGER by and between BANCORPSOUTH BANK and CADENCE BANCORPORATION Dated as of April "
        "12, 2021": "Bancorpsouth Bank / Cadence Bancorporation",
        "AGREEMENT AND PLAN OF MERGER by and among: ADAMAS PHARMACEUTICALS, INC., SUPERNUS PHARMACEUTICALS, INC., and "
        "SUPERNUS REEF, INC. Dated as of October 10, 2021": "Adamas Pharmaceuticals, Inc. / Supernus Pharmaceuticals, Inc.",
    }
    for text, expected in cases.items():
        assert describe(text, "-") == expected
    assert describe("Plain letter with no parties.", "fallback") == "fallback"
