from __future__ import annotations

import json
import unittest

from app.schemas import tool_schemas


class ToolSchemaTests(unittest.TestCase):
    def test_exact_three_tools_and_single_schema_excludes_label(self) -> None:
        schemas = tool_schemas()
        self.assertEqual(
            [item["name"] for item in schemas],
            ["predict_single", "predict_batch", "evaluate_labeled"],
        )
        record = schemas[0]["input_schema"]["properties"]["record"]
        self.assertEqual(len(record["required"]), 25)
        self.assertNotIn("Label", record["properties"])
        self.assertFalse(record["additionalProperties"])
        json.dumps(schemas)


if __name__ == "__main__":
    unittest.main()
