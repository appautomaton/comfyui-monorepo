from __future__ import annotations

import gc
from typing_extensions import override

import torch
import comfy.model_management
from comfy_api.latest import ComfyExtension, io


class ForceFreeVRAMImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ForceFreeVRAMImage",
            display_name="Force Free VRAM (Image)",
            category="utils/memory",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("unload_models", default=True),
                io.Boolean.Input("empty_cache", default=True),
                io.Boolean.Input("run_gc", default=True),
            ],
            outputs=[io.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(
        cls,
        image,
        unload_models=True,
        empty_cache=True,
        run_gc=True,
    ) -> io.NodeOutput:
        if unload_models:
            comfy.model_management.unload_all_models()

        if run_gc:
            gc.collect()

        if empty_cache:
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return io.NodeOutput(image)


class ForceFreeVRAMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ForceFreeVRAMImage]


async def comfy_entrypoint() -> ForceFreeVRAMExtension:
    return ForceFreeVRAMExtension()
