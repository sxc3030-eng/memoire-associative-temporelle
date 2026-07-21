from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.yago_import import (  # noqa: E402
    ImportLimits,
    LINEAGE_ID,
    YAGOImportError,
    import_yago,
    iter_memory_batches,
)


PREFIXES = """\
@prefix yago: <http://yago-knowledge.org/resource/> .
@prefix schema: <http://schema.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
"""

FACTS = PREFIXES + """\
yago:Marie_Curie rdfs:label "Marie Curie"@fr ;
    schema:birthDate "1867-11-07"^^xsd:date .
yago:Marie_Curie schema:birthDate "1868-11-07"^^xsd:date .
yago:Marie_Curie owl:sameAs wd:Q7186 .
yago:Marie_Curie schema:description "Dr. Curie; scientifique."@fr .
"""

META = PREFIXES + """\
<< yago:Marie_Curie schema:birthDate "1867-11-07"^^xsd:date >>
    schema:startDate "1867-11-07T00:00:00Z"^^xsd:dateTime .
<< yago:Marie_Curie schema:birthDate "1867-11-07"^^xsd:date >>
    prov:wasDerivedFrom wd:Q7186 .
"""


class YAGOImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _tiny_zip(self) -> Path:
        archive = self.root / "yago-4.5.0.2-tiny-fixture.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("tiny/yago-facts.ttl", FACTS)
            output.writestr("tiny/yago-meta.ntx", META)
            output.writestr("README.txt", "Fixture locale de test")
        return archive

    def test_streaming_import_preserves_variants_temporal_and_lineage(self) -> None:
        archive = self._tiny_zip()
        database = self.root / "yago-reference.sqlite3"

        report = import_yago(archive, database, strict=True)

        self.assertFalse(report["network_used"])
        self.assertFalse(report["archive_extracted"])
        self.assertEqual(report["dataset"]["license"], "CC BY-SA 3.0")
        self.assertTrue(report["dataset"]["license_url"].endswith("/by-sa/3.0/"))
        self.assertEqual(report["provenance"]["independent_sources"], 1)
        self.assertIn("ne sont jamais", report["provenance"]["warning"])
        self.assertEqual(report["counts"]["claims_created"], 5)
        self.assertEqual(report["counts"]["annotations_created"], 2)
        self.assertEqual(report["counts"]["claims_total"], 5)
        self.assertEqual(report["counts"]["annotations_total"], 2)
        self.assertGreaterEqual(report["counts"]["relations_total"], 4)
        self.assertEqual(report["counts"]["temporal_annotations"], 1)
        self.assertEqual(report["counts"]["provenance_annotations"], 1)
        self.assertGreaterEqual(report["counts"]["variant_groups"], 1)

        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            births = connection.execute(
                """
                SELECT object_value, temporal_start, confidence_level,
                       confidence_basis_json
                FROM claims
                WHERE predicate_iri = 'http://schema.org/birthDate'
                ORDER BY object_value
                """
            ).fetchall()
            self.assertEqual([row["object_value"] for row in births], ["1867-11-07", "1868-11-07"])
            self.assertEqual(births[0]["temporal_start"], "1867-11-07T00:00:00Z")
            self.assertEqual(births[0]["confidence_level"], "statement-attributed")
            basis = json.loads(births[0]["confidence_basis_json"])
            self.assertEqual(basis["lineage_id"], LINEAGE_ID)
            self.assertIsNone(basis["numeric_probability"])
            self.assertIn("Wikidata", basis["lineage"])
        finally:
            connection.close()

    def test_rerun_is_incremental_and_idempotent(self) -> None:
        source = self.root / "facts.ttl"
        source.write_text(FACTS, encoding="utf-8")
        database = self.root / "reference.sqlite3"

        first = import_yago(source, database, strict=True)
        second = import_yago(source, database, strict=True)

        self.assertEqual(first["counts"]["claims_created"], 5)
        self.assertEqual(second["counts"]["claims_created"], 0)
        self.assertEqual(second["counts"]["claim_sources_created"], 0)
        connection = sqlite3.connect(database)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0], 5)
        finally:
            connection.close()

    def test_dry_run_reads_zip_without_creating_database(self) -> None:
        archive = self._tiny_zip()
        database = self.root / "must-not-exist.sqlite3"

        report = import_yago(archive, database, dry_run=True, strict=True)

        self.assertTrue(report["dry_run"])
        self.assertEqual(report["counts"]["triples_seen"], 7)
        self.assertEqual(report["counts"]["claims_created"], 0)
        self.assertFalse(database.exists())

    def test_zip_slip_is_rejected_even_though_nothing_is_extracted(self) -> None:
        archive = self.root / "slip.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape.ttl", FACTS)

        with self.assertRaisesRegex(YAGOImportError, "dangereux"):
            import_yago(archive, dry_run=True)

    def test_suspicious_compression_ratio_is_rejected_before_reading(self) -> None:
        archive = self.root / "bomb.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("facts.ttl", "a" * 100_000)
        limits = replace(ImportLimits(), max_compression_ratio=2.0)

        with self.assertRaisesRegex(YAGOImportError, "Ratio de compression"):
            import_yago(archive, dry_run=True, limits=limits)

    def test_max_triples_stops_with_a_bounded_partial_import(self) -> None:
        source = self.root / "facts.ttl"
        source.write_text(FACTS, encoding="utf-8")
        database = self.root / "limited.sqlite3"
        limits = replace(ImportLimits(), max_triples=2, batch_size=1)

        report = import_yago(source, database, strict=True, limits=limits)

        self.assertTrue(report["counts"]["limited"])
        self.assertEqual(report["counts"]["triples_seen"], 2)
        connection = sqlite3.connect(database)
        try:
            self.assertEqual(connection.execute("SELECT status FROM imports").fetchone()[0], "limited")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0], 2)
        finally:
            connection.close()

    def test_memory_export_is_bounded_and_marks_external_facts_inferred(self) -> None:
        source = self.root / "facts.ttl"
        source.write_text(FACTS, encoding="utf-8")
        database = self.root / "reference.sqlite3"
        import_yago(source, database, strict=True)

        batches = list(iter_memory_batches(database, batch_size=2))

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        record = batches[0][0]
        self.assertEqual(record["source"]["type"], "inferred")
        self.assertEqual(record["source"]["lineage_id"], LINEAGE_ID)
        self.assertTrue(record["idempotency_key"].startswith("yago:"))
        self.assertIn("provenance", record["context"])

    def test_temporal_rdf_star_alone_materialises_a_scoped_claim(self) -> None:
        source = self.root / "meta.ntx"
        source.write_text(META, encoding="utf-8")
        database = self.root / "meta-reference.sqlite3"

        report = import_yago(source, database, strict=True)

        self.assertEqual(report["counts"]["claims_created"], 1)
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT object_value, temporal_start, epistemic_status FROM claims"
            ).fetchone()
            self.assertEqual(row["object_value"], "1867-11-07")
            self.assertEqual(row["temporal_start"], "1867-11-07T00:00:00Z")
            self.assertEqual(row["epistemic_status"], "temporally-scoped-by-rdf-star")
        finally:
            connection.close()

    def test_strict_mode_rejects_unsupported_blank_node(self) -> None:
        source = self.root / "unsupported.ttl"
        source.write_text(PREFIXES + "yago:X schema:knows [ schema:name \"Y\" ] .\n", encoding="utf-8")

        with self.assertRaisesRegex(YAGOImportError, "non pris en charge"):
            import_yago(source, self.root / "reference.sqlite3", strict=True)

    def test_directives_are_bounded_even_before_any_triple(self) -> None:
        source = self.root / "prefixes.ttl"
        source.write_text(
            "@prefix a: <http://a/> .\n@prefix b: <http://b/> .\n@prefix c: <http://c/> .\n",
            encoding="utf-8",
        )
        limits = replace(ImportLimits(), max_statements=2)

        with self.assertRaisesRegex(YAGOImportError, "instructions Turtle"):
            import_yago(source, dry_run=True, limits=limits)


if __name__ == "__main__":
    unittest.main()
