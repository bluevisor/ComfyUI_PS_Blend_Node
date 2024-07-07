# ComfyUI PS Blend Node

This repository contains a custom node for ComfyUI that implements familiar PS-style blend modes using PyTorch. The `PSBlendNode` allows you to blend two images together using a variety of blend modes and an opacity parameter.

## Features

- Supports multiple blend modes including Normal, Darken, Multiply, Color Burn, Linear Burn, Lighten, Screen, Color Dodge, Linear Dodge, Overlay, Soft Light, Hard Light, Difference, Exclusion, Subtract, Divide, Hue, Saturation, Color, and Luminosity.
- Blends images with an adjustable opacity setting.
- Handles images with alpha channels.

## Installation

To install this node, clone the repository into the `custom_nodes` directory of your ComfyUI installation:

```bash
cd path/to/ComfyUI/custom_nodes
git clone https://github.com/bluevisor/ComfyUI_PS_Blend_Node.git PSBlendNode
```
## Usage
- To use the PSBlendNode, follow these steps:
-- Load two images into your ComfyUI workflow.
-- Connect the images to the PSBlendNode input.
-- Select a blend mode.
-- Blend away.
