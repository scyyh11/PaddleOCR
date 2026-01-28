# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton Python Backend for restructure-pages operation.

TODO: This model will be included in the HPS SDK in the future.
      Remove this file and the model_repository/restructure-pages directory
      once the SDK includes this model.
"""

import json

import numpy as np
import triton_python_backend_utils as pb_utils

from paddlex.inference.pipelines.layout_parsing.merge_table import (
    merge_tables_across_pages,
)
from paddlex.inference.pipelines.layout_parsing.title_level import (
    assign_levels_to_parsing_res,
)


class TritonPythonModel:
    """Triton Python model for restructuring multi-page parsing results."""

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])
        self.logger = pb_utils.Logger

    def execute(self, requests):
        """Execute restructure-pages requests."""
        responses = []

        for request in requests:
            try:
                # Parse input JSON
                input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
                input_json = input_tensor.as_numpy()[0].decode("utf-8")
                body = json.loads(input_json)

                # Extract parameters
                pages = body.get("pages", [])
                merge_tables = body.get("mergeTables", True)
                relevel_titles = body.get("relevelTitles", True)
                concatenate = body.get("concatenatePages", False)
                log_id = body.get("logId", "")

                # Collect all parsing results by page
                blocks_by_page = []
                all_layout_det_res = []
                all_images = {}

                for idx, page in enumerate(pages):
                    pruned_result = page.get("prunedResult", {})
                    parsing_res = pruned_result.get("parsing_res_list", [])
                    blocks_by_page.append(parsing_res)

                    # Collect layout detection results if available
                    layout_det = pruned_result.get("layout_det_res", [])
                    if isinstance(layout_det, list):
                        all_layout_det_res.extend(layout_det)
                    else:
                        all_layout_det_res.append(layout_det)

                    # Collect markdown images
                    md_images = page.get("markdownImages", {}) or {}
                    for key, value in md_images.items():
                        new_key = f"page{idx}_{key}"
                        all_images[new_key] = value

                # Apply merge tables logic
                if merge_tables:
                    blocks_by_page = merge_tables_across_pages(blocks_by_page)

                # Apply title releveling logic
                if relevel_titles:
                    blocks_by_page = assign_levels_to_parsing_res(
                        blocks_by_page, all_layout_det_res
                    )

                # Flatten blocks
                all_parsing_res = []
                for page_blocks in blocks_by_page:
                    all_parsing_res.extend(page_blocks)

                # Build combined result
                combined_result = {
                    "parsing_res_list": all_parsing_res,
                }

                # Generate markdown if concatenation requested
                markdown_data = None
                if concatenate:
                    markdown_text = self._generate_markdown(all_parsing_res)
                    markdown_data = {
                        "text": markdown_text,
                        "images": all_images if all_images else None,
                    }

                # Build response
                result = {
                    "logId": log_id,
                    "errorCode": 0,
                    "errorMsg": "Success",
                    "result": {
                        "layoutParsingResult": {
                            "prunedResult": combined_result,
                            "markdown": (
                                markdown_data
                                if markdown_data
                                else {"text": "", "images": None}
                            ),
                        }
                    },
                }

                output_json = json.dumps(result)
                output_tensor = pb_utils.Tensor(
                    "OUTPUT",
                    np.array([output_json.encode("utf-8")], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse([output_tensor]))

            except Exception as e:
                self.logger.log_error(f"Error in restructure-pages: {e}")
                error_result = {
                    "logId": body.get("logId", "") if "body" in dir() else "",
                    "errorCode": 500,
                    "errorMsg": str(e),
                }
                output_json = json.dumps(error_result)
                output_tensor = pb_utils.Tensor(
                    "OUTPUT",
                    np.array([output_json.encode("utf-8")], dtype=object),
                )
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def _generate_markdown(self, parsing_res_list: list) -> str:
        """Generate markdown text from parsing result blocks."""
        markdown_parts = []
        for block in parsing_res_list:
            content = block.get("block_content", "")
            if content:
                markdown_parts.append(content)
        return "\n\n".join(markdown_parts)

    def finalize(self):
        """Clean up resources."""
        pass
