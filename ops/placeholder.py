from typing import Any, Tuple

import torch
import triton
import triton.language as tl

SIZE_BLOCKS_POS = 4
SIZE_BLOCKS_CHANNELS = 128
NUM_WARPS = 16
NUM_STAGES = 1


@triton.jit
def geometric_product_kernel_forward(
    ptr_x: tl.tensor,
    ptr_y: tl.tensor,
    ptr_outputs: tl.tensor,
    num_pos: tl.constexpr,
    num_channels: tl.constexpr,
    size_blocks_pos: tl.constexpr,
    size_blocks_channels: tl.constexpr,
) -> None:

    offsets_pos = tl.program_id(axis=0) * size_blocks_pos + tl.arange(0, size_blocks_pos)
    offsets_channels = tl.program_id(axis=1) * size_blocks_channels + tl.arange(
        0, size_blocks_channels
    )

    offset = offsets_pos[:, None] * num_channels + offsets_channels[None, :]
    stride = num_pos * num_channels

    mask = (offsets_pos < num_pos)[:, None] & (offsets_channels < num_channels)[None, :]

    x0 = tl.load(ptr_x + offset + 0 * stride, mask)
    x1 = tl.load(ptr_x + offset + 1 * stride, mask)
    x2 = tl.load(ptr_x + offset + 2 * stride, mask)
    x3 = tl.load(ptr_x + offset + 3 * stride, mask)
    x4 = tl.load(ptr_x + offset + 4 * stride, mask)
    x5 = tl.load(ptr_x + offset + 5 * stride, mask)
    x6 = tl.load(ptr_x + offset + 6 * stride, mask)
    x7 = tl.load(ptr_x + offset + 7 * stride, mask)
    x8 = tl.load(ptr_x + offset + 8 * stride, mask)
    x9 = tl.load(ptr_x + offset + 9 * stride, mask)
    x10 = tl.load(ptr_x + offset + 10 * stride, mask)
    x11 = tl.load(ptr_x + offset + 11 * stride, mask)
    x12 = tl.load(ptr_x + offset + 12 * stride, mask)
    x13 = tl.load(ptr_x + offset + 13 * stride, mask)
    x14 = tl.load(ptr_x + offset + 14 * stride, mask)
    x15 = tl.load(ptr_x + offset + 15 * stride, mask)

    y0 = tl.load(ptr_y + offset + 0 * stride, mask)
    y1 = tl.load(ptr_y + offset + 1 * stride, mask)
    y2 = tl.load(ptr_y + offset + 2 * stride, mask)
    y3 = tl.load(ptr_y + offset + 3 * stride, mask)
    y4 = tl.load(ptr_y + offset + 4 * stride, mask)
    y5 = tl.load(ptr_y + offset + 5 * stride, mask)
    y6 = tl.load(ptr_y + offset + 6 * stride, mask)
    y7 = tl.load(ptr_y + offset + 7 * stride, mask)
    y8 = tl.load(ptr_y + offset + 8 * stride, mask)
    y9 = tl.load(ptr_y + offset + 9 * stride, mask)
    y10 = tl.load(ptr_y + offset + 10 * stride, mask)
    y11 = tl.load(ptr_y + offset + 11 * stride, mask)
    y12 = tl.load(ptr_y + offset + 12 * stride, mask)
    y13 = tl.load(ptr_y + offset + 13 * stride, mask)
    y14 = tl.load(ptr_y + offset + 14 * stride, mask)
    y15 = tl.load(ptr_y + offset + 15 * stride, mask)

    outputs0 = x0 * y0 + x2 * y2 + x3 * y3 + x4 * y4 - x8 * y8 - x9 * y9 - x10 * y10 - x14 * y14
    outputs1 = (
        x0 * y1
        + x1 * y0
        - x2 * y5
        - x3 * y6
        - x4 * y7
        + x5 * y2
        + x6 * y3
        + x7 * y4
        - x8 * y11
        - x9 * y12
        - x10 * y13
        - x11 * y8
        - x12 * y9
        - x13 * y10
        + x14 * y15
        - x15 * y14
    )
    outputs2 = x0 * y2 + x2 * y0 - x3 * y8 - x4 * y9 + x8 * y3 + x9 * y4 - x10 * y14 - x14 * y10
    outputs3 = x0 * y3 + x2 * y8 + x3 * y0 - x4 * y10 - x8 * y2 + x9 * y14 + x10 * y4 + x14 * y9
    outputs4 = x0 * y4 + x2 * y9 + x3 * y10 + x4 * y0 - x8 * y14 - x9 * y2 - x10 * y3 - x14 * y8
    outputs5 = (
        x0 * y5
        + x1 * y2
        - x2 * y1
        + x3 * y11
        + x4 * y12
        + x5 * y0
        - x6 * y8
        - x7 * y9
        + x8 * y6
        + x9 * y7
        - x10 * y15
        + x11 * y3
        + x12 * y4
        - x13 * y14
        + x14 * y13
        - x15 * y10
    )
    outputs6 = (
        x0 * y6
        + x1 * y3
        - x2 * y11
        - x3 * y1
        + x4 * y13
        + x5 * y8
        + x6 * y0
        - x7 * y10
        - x8 * y5
        + x9 * y15
        + x10 * y7
        - x11 * y2
        + x12 * y14
        + x13 * y4
        - x14 * y12
        + x15 * y9
    )
    outputs7 = (
        x0 * y7
        + x1 * y4
        - x2 * y12
        - x3 * y13
        - x4 * y1
        + x5 * y9
        + x6 * y10
        + x7 * y0
        - x8 * y15
        - x9 * y5
        - x10 * y6
        - x11 * y14
        - x12 * y2
        - x13 * y3
        + x14 * y11
        - x15 * y8
    )
    outputs8 = x0 * y8 + x2 * y3 - x3 * y2 + x4 * y14 + x8 * y0 - x9 * y10 + x10 * y9 + x14 * y4
    outputs9 = x0 * y9 + x2 * y4 - x3 * y14 - x4 * y2 + x8 * y10 + x9 * y0 - x10 * y8 - x14 * y3
    outputs10 = x0 * y10 + x2 * y14 + x3 * y4 - x4 * y3 - x8 * y9 + x9 * y8 + x10 * y0 + x14 * y2
    outputs11 = (
        x0 * y11
        + x1 * y8
        - x2 * y6
        + x3 * y5
        - x4 * y15
        + x5 * y3
        - x6 * y2
        + x7 * y14
        + x8 * y1
        - x9 * y13
        + x10 * y12
        + x11 * y0
        - x12 * y10
        + x13 * y9
        - x14 * y7
        + x15 * y4
    )
    outputs12 = (
        x0 * y12
        + x1 * y9
        - x2 * y7
        + x3 * y15
        + x4 * y5
        + x5 * y4
        - x6 * y14
        - x7 * y2
        + x8 * y13
        + x9 * y1
        - x10 * y11
        + x11 * y10
        + x12 * y0
        - x13 * y8
        + x14 * y6
        - x15 * y3
    )
    outputs13 = (
        x0 * y13
        + x1 * y10
        - x2 * y15
        - x3 * y7
        + x4 * y6
        + x5 * y14
        + x6 * y4
        - x7 * y3
        - x8 * y12
        + x9 * y11
        + x10 * y1
        - x11 * y9
        + x12 * y8
        + x13 * y0
        - x14 * y5
        + x15 * y2
    )
    outputs14 = x0 * y14 + x2 * y10 - x3 * y9 + x4 * y8 + x8 * y4 - x9 * y3 + x10 * y2 + x14 * y0
    outputs15 = (
        x0 * y15
        + x1 * y14
        - x2 * y13
        + x3 * y12
        - x4 * y11
        + x5 * y10
        - x6 * y9
        + x7 * y8
        + x8 * y7
        - x9 * y6
        + x10 * y5
        + x11 * y4
        - x12 * y3
        + x13 * y2
        - x14 * y1
        + x15 * y0
    )

    tl.store(ptr_outputs + offset + 0 * stride, outputs0, mask)
    tl.store(ptr_outputs + offset + 1 * stride, outputs1, mask)
    tl.store(ptr_outputs + offset + 2 * stride, outputs2, mask)
    tl.store(ptr_outputs + offset + 3 * stride, outputs3, mask)
    tl.store(ptr_outputs + offset + 4 * stride, outputs4, mask)
    tl.store(ptr_outputs + offset + 5 * stride, outputs5, mask)
    tl.store(ptr_outputs + offset + 6 * stride, outputs6, mask)
    tl.store(ptr_outputs + offset + 7 * stride, outputs7, mask)
    tl.store(ptr_outputs + offset + 8 * stride, outputs8, mask)
    tl.store(ptr_outputs + offset + 9 * stride, outputs9, mask)
    tl.store(ptr_outputs + offset + 10 * stride, outputs10, mask)
    tl.store(ptr_outputs + offset + 11 * stride, outputs11, mask)
    tl.store(ptr_outputs + offset + 12 * stride, outputs12, mask)
    tl.store(ptr_outputs + offset + 13 * stride, outputs13, mask)
    tl.store(ptr_outputs + offset + 14 * stride, outputs14, mask)
    tl.store(ptr_outputs + offset + 15 * stride, outputs15, mask)


@triton.jit
def geometric_product_kernel_backward(
    ptr_x: tl.tensor,
    ptr_y: tl.tensor,
    ptr_grad_x: tl.tensor,
    ptr_grad_y: tl.tensor,
    ptr_grad_outputs: tl.tensor,
    num_pos: tl.constexpr,
    num_channels: tl.constexpr,
    size_blocks_pos: tl.constexpr,
    size_blocks_channels: tl.constexpr,
) -> None:

    offsets_pos = tl.program_id(axis=0) * size_blocks_pos + tl.arange(0, size_blocks_pos)
    offsets_channels = tl.program_id(axis=1) * size_blocks_channels + tl.arange(
        0, size_blocks_channels
    )

    offset = offsets_pos[:, None] * num_channels + offsets_channels[None, :]
    stride = num_pos * num_channels

    mask = (offsets_pos < num_pos)[:, None] & (offsets_channels < num_channels)[None, :]

    x0 = tl.load(ptr_x + offset + 0 * stride, mask)
    x1 = tl.load(ptr_x + offset + 1 * stride, mask)
    x2 = tl.load(ptr_x + offset + 2 * stride, mask)
    x3 = tl.load(ptr_x + offset + 3 * stride, mask)
    x4 = tl.load(ptr_x + offset + 4 * stride, mask)
    x5 = tl.load(ptr_x + offset + 5 * stride, mask)
    x6 = tl.load(ptr_x + offset + 6 * stride, mask)
    x7 = tl.load(ptr_x + offset + 7 * stride, mask)
    x8 = tl.load(ptr_x + offset + 8 * stride, mask)
    x9 = tl.load(ptr_x + offset + 9 * stride, mask)
    x10 = tl.load(ptr_x + offset + 10 * stride, mask)
    x11 = tl.load(ptr_x + offset + 11 * stride, mask)
    x12 = tl.load(ptr_x + offset + 12 * stride, mask)
    x13 = tl.load(ptr_x + offset + 13 * stride, mask)
    x14 = tl.load(ptr_x + offset + 14 * stride, mask)
    x15 = tl.load(ptr_x + offset + 15 * stride, mask)

    y0 = tl.load(ptr_y + offset + 0 * stride, mask)
    y1 = tl.load(ptr_y + offset + 1 * stride, mask)
    y2 = tl.load(ptr_y + offset + 2 * stride, mask)
    y3 = tl.load(ptr_y + offset + 3 * stride, mask)
    y4 = tl.load(ptr_y + offset + 4 * stride, mask)
    y5 = tl.load(ptr_y + offset + 5 * stride, mask)
    y6 = tl.load(ptr_y + offset + 6 * stride, mask)
    y7 = tl.load(ptr_y + offset + 7 * stride, mask)
    y8 = tl.load(ptr_y + offset + 8 * stride, mask)
    y9 = tl.load(ptr_y + offset + 9 * stride, mask)
    y10 = tl.load(ptr_y + offset + 10 * stride, mask)
    y11 = tl.load(ptr_y + offset + 11 * stride, mask)
    y12 = tl.load(ptr_y + offset + 12 * stride, mask)
    y13 = tl.load(ptr_y + offset + 13 * stride, mask)
    y14 = tl.load(ptr_y + offset + 14 * stride, mask)
    y15 = tl.load(ptr_y + offset + 15 * stride, mask)

    grad_outputs0 = tl.load(ptr_grad_outputs + offset + 0 * stride, mask)
    grad_outputs1 = tl.load(ptr_grad_outputs + offset + 1 * stride, mask)
    grad_outputs2 = tl.load(ptr_grad_outputs + offset + 2 * stride, mask)
    grad_outputs3 = tl.load(ptr_grad_outputs + offset + 3 * stride, mask)
    grad_outputs4 = tl.load(ptr_grad_outputs + offset + 4 * stride, mask)
    grad_outputs5 = tl.load(ptr_grad_outputs + offset + 5 * stride, mask)
    grad_outputs6 = tl.load(ptr_grad_outputs + offset + 6 * stride, mask)
    grad_outputs7 = tl.load(ptr_grad_outputs + offset + 7 * stride, mask)
    grad_outputs8 = tl.load(ptr_grad_outputs + offset + 8 * stride, mask)
    grad_outputs9 = tl.load(ptr_grad_outputs + offset + 9 * stride, mask)
    grad_outputs10 = tl.load(ptr_grad_outputs + offset + 10 * stride, mask)
    grad_outputs11 = tl.load(ptr_grad_outputs + offset + 11 * stride, mask)
    grad_outputs12 = tl.load(ptr_grad_outputs + offset + 12 * stride, mask)
    grad_outputs13 = tl.load(ptr_grad_outputs + offset + 13 * stride, mask)
    grad_outputs14 = tl.load(ptr_grad_outputs + offset + 14 * stride, mask)
    grad_outputs15 = tl.load(ptr_grad_outputs + offset + 15 * stride, mask)

    grad_x0 = (
        grad_outputs0 * y0
        + grad_outputs1 * y1
        + grad_outputs2 * y2
        + grad_outputs3 * y3
        + grad_outputs4 * y4
        + grad_outputs5 * y5
        + grad_outputs6 * y6
        + grad_outputs7 * y7
        + grad_outputs8 * y8
        + grad_outputs9 * y9
        + grad_outputs10 * y10
        + grad_outputs11 * y11
        + grad_outputs12 * y12
        + grad_outputs13 * y13
        + grad_outputs14 * y14
        + grad_outputs15 * y15
    )
    grad_x1 = (
        grad_outputs1 * y0
        + grad_outputs5 * y2
        + grad_outputs6 * y3
        + grad_outputs7 * y4
        + grad_outputs11 * y8
        + grad_outputs12 * y9
        + grad_outputs13 * y10
        + grad_outputs15 * y14
    )
    grad_x2 = (
        grad_outputs0 * y2
        - grad_outputs1 * y5
        + grad_outputs2 * y0
        + grad_outputs3 * y8
        + grad_outputs4 * y9
        - grad_outputs5 * y1
        - grad_outputs6 * y11
        - grad_outputs7 * y12
        + grad_outputs8 * y3
        + grad_outputs9 * y4
        + grad_outputs10 * y14
        - grad_outputs11 * y6
        - grad_outputs12 * y7
        - grad_outputs13 * y15
        + grad_outputs14 * y10
        - grad_outputs15 * y13
    )
    grad_x3 = (
        grad_outputs0 * y3
        - grad_outputs1 * y6
        - grad_outputs2 * y8
        + grad_outputs3 * y0
        + grad_outputs4 * y10
        + grad_outputs5 * y11
        - grad_outputs6 * y1
        - grad_outputs7 * y13
        - grad_outputs8 * y2
        - grad_outputs9 * y14
        + grad_outputs10 * y4
        + grad_outputs11 * y5
        + grad_outputs12 * y15
        - grad_outputs13 * y7
        - grad_outputs14 * y9
        + grad_outputs15 * y12
    )
    grad_x4 = (
        grad_outputs0 * y4
        - grad_outputs1 * y7
        - grad_outputs2 * y9
        - grad_outputs3 * y10
        + grad_outputs4 * y0
        + grad_outputs5 * y12
        + grad_outputs6 * y13
        - grad_outputs7 * y1
        + grad_outputs8 * y14
        - grad_outputs9 * y2
        - grad_outputs10 * y3
        - grad_outputs11 * y15
        + grad_outputs12 * y5
        + grad_outputs13 * y6
        + grad_outputs14 * y8
        - grad_outputs15 * y11
    )
    grad_x5 = (
        grad_outputs1 * y2
        + grad_outputs5 * y0
        + grad_outputs6 * y8
        + grad_outputs7 * y9
        + grad_outputs11 * y3
        + grad_outputs12 * y4
        + grad_outputs13 * y14
        + grad_outputs15 * y10
    )
    grad_x6 = (
        grad_outputs1 * y3
        - grad_outputs5 * y8
        + grad_outputs6 * y0
        + grad_outputs7 * y10
        - grad_outputs11 * y2
        - grad_outputs12 * y14
        + grad_outputs13 * y4
        - grad_outputs15 * y9
    )
    grad_x7 = (
        grad_outputs1 * y4
        - grad_outputs5 * y9
        - grad_outputs6 * y10
        + grad_outputs7 * y0
        + grad_outputs11 * y14
        - grad_outputs12 * y2
        - grad_outputs13 * y3
        + grad_outputs15 * y8
    )
    grad_x8 = (
        -grad_outputs0 * y8
        - grad_outputs1 * y11
        + grad_outputs2 * y3
        - grad_outputs3 * y2
        - grad_outputs4 * y14
        + grad_outputs5 * y6
        - grad_outputs6 * y5
        - grad_outputs7 * y15
        + grad_outputs8 * y0
        + grad_outputs9 * y10
        - grad_outputs10 * y9
        + grad_outputs11 * y1
        + grad_outputs12 * y13
        - grad_outputs13 * y12
        + grad_outputs14 * y4
        + grad_outputs15 * y7
    )
    grad_x9 = (
        -grad_outputs0 * y9
        - grad_outputs1 * y12
        + grad_outputs2 * y4
        + grad_outputs3 * y14
        - grad_outputs4 * y2
        + grad_outputs5 * y7
        + grad_outputs6 * y15
        - grad_outputs7 * y5
        - grad_outputs8 * y10
        + grad_outputs9 * y0
        + grad_outputs10 * y8
        - grad_outputs11 * y13
        + grad_outputs12 * y1
        + grad_outputs13 * y11
        - grad_outputs14 * y3
        - grad_outputs15 * y6
    )
    grad_x10 = (
        -grad_outputs0 * y10
        - grad_outputs1 * y13
        - grad_outputs2 * y14
        + grad_outputs3 * y4
        - grad_outputs4 * y3
        - grad_outputs5 * y15
        + grad_outputs6 * y7
        - grad_outputs7 * y6
        + grad_outputs8 * y9
        - grad_outputs9 * y8
        + grad_outputs10 * y0
        + grad_outputs11 * y12
        - grad_outputs12 * y11
        + grad_outputs13 * y1
        + grad_outputs14 * y2
        + grad_outputs15 * y5
    )
    grad_x11 = (
        -grad_outputs1 * y8
        + grad_outputs5 * y3
        - grad_outputs6 * y2
        - grad_outputs7 * y14
        + grad_outputs11 * y0
        + grad_outputs12 * y10
        - grad_outputs13 * y9
        + grad_outputs15 * y4
    )
    grad_x12 = (
        -grad_outputs1 * y9
        + grad_outputs5 * y4
        + grad_outputs6 * y14
        - grad_outputs7 * y2
        - grad_outputs11 * y10
        + grad_outputs12 * y0
        + grad_outputs13 * y8
        - grad_outputs15 * y3
    )
    grad_x13 = (
        -grad_outputs1 * y10
        - grad_outputs5 * y14
        + grad_outputs6 * y4
        - grad_outputs7 * y3
        + grad_outputs11 * y9
        - grad_outputs12 * y8
        + grad_outputs13 * y0
        + grad_outputs15 * y2
    )
    grad_x14 = (
        -grad_outputs0 * y14
        + grad_outputs1 * y15
        - grad_outputs2 * y10
        + grad_outputs3 * y9
        - grad_outputs4 * y8
        + grad_outputs5 * y13
        - grad_outputs6 * y12
        + grad_outputs7 * y11
        + grad_outputs8 * y4
        - grad_outputs9 * y3
        + grad_outputs10 * y2
        - grad_outputs11 * y7
        + grad_outputs12 * y6
        - grad_outputs13 * y5
        + grad_outputs14 * y0
        - grad_outputs15 * y1
    )
    grad_x15 = (
        -grad_outputs1 * y14
        - grad_outputs5 * y10
        + grad_outputs6 * y9
        - grad_outputs7 * y8
        + grad_outputs11 * y4
        - grad_outputs12 * y3
        + grad_outputs13 * y2
        + grad_outputs15 * y0
    )

    grad_y0 = (
        grad_outputs0 * x0
        + grad_outputs1 * x1
        + grad_outputs2 * x2
        + grad_outputs3 * x3
        + grad_outputs4 * x4
        + grad_outputs5 * x5
        + grad_outputs6 * x6
        + grad_outputs7 * x7
        + grad_outputs8 * x8
        + grad_outputs9 * x9
        + grad_outputs10 * x10
        + grad_outputs11 * x11
        + grad_outputs12 * x12
        + grad_outputs13 * x13
        + grad_outputs14 * x14
        + grad_outputs15 * x15
    )
    grad_y1 = (
        grad_outputs1 * x0
        - grad_outputs5 * x2
        - grad_outputs6 * x3
        - grad_outputs7 * x4
        + grad_outputs11 * x8
        + grad_outputs12 * x9
        + grad_outputs13 * x10
        - grad_outputs15 * x14
    )
    grad_y2 = (
        grad_outputs0 * x2
        + grad_outputs1 * x5
        + grad_outputs2 * x0
        - grad_outputs3 * x8
        - grad_outputs4 * x9
        + grad_outputs5 * x1
        - grad_outputs6 * x11
        - grad_outputs7 * x12
        - grad_outputs8 * x3
        - grad_outputs9 * x4
        + grad_outputs10 * x14
        - grad_outputs11 * x6
        - grad_outputs12 * x7
        + grad_outputs13 * x15
        + grad_outputs14 * x10
        + grad_outputs15 * x13
    )
    grad_y3 = (
        grad_outputs0 * x3
        + grad_outputs1 * x6
        + grad_outputs2 * x8
        + grad_outputs3 * x0
        - grad_outputs4 * x10
        + grad_outputs5 * x11
        + grad_outputs6 * x1
        - grad_outputs7 * x13
        + grad_outputs8 * x2
        - grad_outputs9 * x14
        - grad_outputs10 * x4
        + grad_outputs11 * x5
        - grad_outputs12 * x15
        - grad_outputs13 * x7
        - grad_outputs14 * x9
        - grad_outputs15 * x12
    )
    grad_y4 = (
        grad_outputs0 * x4
        + grad_outputs1 * x7
        + grad_outputs2 * x9
        + grad_outputs3 * x10
        + grad_outputs4 * x0
        + grad_outputs5 * x12
        + grad_outputs6 * x13
        + grad_outputs7 * x1
        + grad_outputs8 * x14
        + grad_outputs9 * x2
        + grad_outputs10 * x3
        + grad_outputs11 * x15
        + grad_outputs12 * x5
        + grad_outputs13 * x6
        + grad_outputs14 * x8
        + grad_outputs15 * x11
    )
    grad_y5 = (
        -grad_outputs1 * x2
        + grad_outputs5 * x0
        - grad_outputs6 * x8
        - grad_outputs7 * x9
        + grad_outputs11 * x3
        + grad_outputs12 * x4
        - grad_outputs13 * x14
        + grad_outputs15 * x10
    )
    grad_y6 = (
        -grad_outputs1 * x3
        + grad_outputs5 * x8
        + grad_outputs6 * x0
        - grad_outputs7 * x10
        - grad_outputs11 * x2
        + grad_outputs12 * x14
        + grad_outputs13 * x4
        - grad_outputs15 * x9
    )
    grad_y7 = (
        -grad_outputs1 * x4
        + grad_outputs5 * x9
        + grad_outputs6 * x10
        + grad_outputs7 * x0
        - grad_outputs11 * x14
        - grad_outputs12 * x2
        - grad_outputs13 * x3
        + grad_outputs15 * x8
    )
    grad_y8 = (
        -grad_outputs0 * x8
        - grad_outputs1 * x11
        - grad_outputs2 * x3
        + grad_outputs3 * x2
        - grad_outputs4 * x14
        - grad_outputs5 * x6
        + grad_outputs6 * x5
        - grad_outputs7 * x15
        + grad_outputs8 * x0
        - grad_outputs9 * x10
        + grad_outputs10 * x9
        + grad_outputs11 * x1
        - grad_outputs12 * x13
        + grad_outputs13 * x12
        + grad_outputs14 * x4
        + grad_outputs15 * x7
    )
    grad_y9 = (
        -grad_outputs0 * x9
        - grad_outputs1 * x12
        - grad_outputs2 * x4
        + grad_outputs3 * x14
        + grad_outputs4 * x2
        - grad_outputs5 * x7
        + grad_outputs6 * x15
        + grad_outputs7 * x5
        + grad_outputs8 * x10
        + grad_outputs9 * x0
        - grad_outputs10 * x8
        + grad_outputs11 * x13
        + grad_outputs12 * x1
        - grad_outputs13 * x11
        - grad_outputs14 * x3
        - grad_outputs15 * x6
    )
    grad_y10 = (
        -grad_outputs0 * x10
        - grad_outputs1 * x13
        - grad_outputs2 * x14
        - grad_outputs3 * x4
        + grad_outputs4 * x3
        - grad_outputs5 * x15
        - grad_outputs6 * x7
        + grad_outputs7 * x6
        - grad_outputs8 * x9
        + grad_outputs9 * x8
        + grad_outputs10 * x0
        - grad_outputs11 * x12
        + grad_outputs12 * x11
        + grad_outputs13 * x1
        + grad_outputs14 * x2
        + grad_outputs15 * x5
    )
    grad_y11 = (
        -grad_outputs1 * x8
        + grad_outputs5 * x3
        - grad_outputs6 * x2
        + grad_outputs7 * x14
        + grad_outputs11 * x0
        - grad_outputs12 * x10
        + grad_outputs13 * x9
        - grad_outputs15 * x4
    )
    grad_y12 = (
        -grad_outputs1 * x9
        + grad_outputs5 * x4
        - grad_outputs6 * x14
        - grad_outputs7 * x2
        + grad_outputs11 * x10
        + grad_outputs12 * x0
        - grad_outputs13 * x8
        + grad_outputs15 * x3
    )
    grad_y13 = (
        -grad_outputs1 * x10
        + grad_outputs5 * x14
        + grad_outputs6 * x4
        - grad_outputs7 * x3
        - grad_outputs11 * x9
        + grad_outputs12 * x8
        + grad_outputs13 * x0
        - grad_outputs15 * x2
    )
    grad_y14 = (
        -grad_outputs0 * x14
        - grad_outputs1 * x15
        - grad_outputs2 * x10
        + grad_outputs3 * x9
        - grad_outputs4 * x8
        - grad_outputs5 * x13
        + grad_outputs6 * x12
        - grad_outputs7 * x11
        + grad_outputs8 * x4
        - grad_outputs9 * x3
        + grad_outputs10 * x2
        + grad_outputs11 * x7
        - grad_outputs12 * x6
        + grad_outputs13 * x5
        + grad_outputs14 * x0
        + grad_outputs15 * x1
    )
    grad_y15 = (
        grad_outputs1 * x14
        - grad_outputs5 * x10
        + grad_outputs6 * x9
        - grad_outputs7 * x8
        - grad_outputs11 * x4
        + grad_outputs12 * x3
        - grad_outputs13 * x2
        + grad_outputs15 * x0
    )

    tl.store(ptr_grad_x + offset + 0 * stride, grad_x0, mask)
    tl.store(ptr_grad_x + offset + 1 * stride, grad_x1, mask)
    tl.store(ptr_grad_x + offset + 2 * stride, grad_x2, mask)
    tl.store(ptr_grad_x + offset + 3 * stride, grad_x3, mask)
    tl.store(ptr_grad_x + offset + 4 * stride, grad_x4, mask)
    tl.store(ptr_grad_x + offset + 5 * stride, grad_x5, mask)
    tl.store(ptr_grad_x + offset + 6 * stride, grad_x6, mask)
    tl.store(ptr_grad_x + offset + 7 * stride, grad_x7, mask)
    tl.store(ptr_grad_x + offset + 8 * stride, grad_x8, mask)
    tl.store(ptr_grad_x + offset + 9 * stride, grad_x9, mask)
    tl.store(ptr_grad_x + offset + 10 * stride, grad_x10, mask)
    tl.store(ptr_grad_x + offset + 11 * stride, grad_x11, mask)
    tl.store(ptr_grad_x + offset + 12 * stride, grad_x12, mask)
    tl.store(ptr_grad_x + offset + 13 * stride, grad_x13, mask)
    tl.store(ptr_grad_x + offset + 14 * stride, grad_x14, mask)
    tl.store(ptr_grad_x + offset + 15 * stride, grad_x15, mask)

    tl.store(ptr_grad_y + offset + 0 * stride, grad_y0, mask)
    tl.store(ptr_grad_y + offset + 1 * stride, grad_y1, mask)
    tl.store(ptr_grad_y + offset + 2 * stride, grad_y2, mask)
    tl.store(ptr_grad_y + offset + 3 * stride, grad_y3, mask)
    tl.store(ptr_grad_y + offset + 4 * stride, grad_y4, mask)
    tl.store(ptr_grad_y + offset + 5 * stride, grad_y5, mask)
    tl.store(ptr_grad_y + offset + 6 * stride, grad_y6, mask)
    tl.store(ptr_grad_y + offset + 7 * stride, grad_y7, mask)
    tl.store(ptr_grad_y + offset + 8 * stride, grad_y8, mask)
    tl.store(ptr_grad_y + offset + 9 * stride, grad_y9, mask)
    tl.store(ptr_grad_y + offset + 10 * stride, grad_y10, mask)
    tl.store(ptr_grad_y + offset + 11 * stride, grad_y11, mask)
    tl.store(ptr_grad_y + offset + 12 * stride, grad_y12, mask)
    tl.store(ptr_grad_y + offset + 13 * stride, grad_y13, mask)
    tl.store(ptr_grad_y + offset + 14 * stride, grad_y14, mask)
    tl.store(ptr_grad_y + offset + 15 * stride, grad_y15, mask)


def geometric_product_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Dimensions mismatch."

    outputs = torch.empty_like(x)

    num_dim, num_pos, num_channels = outputs.shape
    assert num_dim == 16, "Only G(3, 0, 1) multivectors supported."

    size_blocks_pos = min(SIZE_BLOCKS_POS, num_pos)
    size_blocks_channels = min(SIZE_BLOCKS_CHANNELS, num_channels)

    num_blocks_pos = triton.cdiv(num_pos, size_blocks_pos)
    num_blocks_channels = triton.cdiv(num_channels, size_blocks_channels)

    grid = (num_blocks_pos, num_blocks_channels)
    geometric_product_kernel_forward[grid](
        x,
        y,
        outputs,
        num_pos,
        num_channels,
        size_blocks_pos,
        size_blocks_channels,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return outputs


def geometric_product_backward(
    x: torch.Tensor, y: torch.Tensor, grad_outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_dim, num_pos, num_channels = grad_outputs.shape

    size_blocks_pos = min(SIZE_BLOCKS_POS, num_pos)
    size_blocks_channels = min(SIZE_BLOCKS_CHANNELS, num_channels)

    num_blocks_pos = triton.cdiv(num_pos, size_blocks_pos)
    num_blocks_channels = triton.cdiv(num_channels, size_blocks_channels)

    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)

    grid = (num_blocks_pos, num_blocks_channels)
    geometric_product_kernel_backward[grid](
        x,
        y,
        grad_x,
        grad_y,
        grad_outputs,
        num_pos,
        num_channels,
        size_blocks_pos,
        size_blocks_channels,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return grad_x, grad_y


class GeometricProduct(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous(), "Tensors required to be contiguous."

        outputs = geometric_product_forward(x, y)
        ctx.save_for_backward(x, y)

        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_outputs = grad_outputs.contiguous()

        x, y = ctx.saved_tensors
        grad_x, grad_y = geometric_product_backward(x, y, grad_outputs)

        return grad_x, grad_y


def placeholder(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return GeometricProduct.apply(x, y)
