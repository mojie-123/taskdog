# Local model checkpoints

`pointbert/PointTransformer_ModelNet8192points.pth` is the Point-BERT
ModelNet40 8192-point checkpoint used to initialize the N2M encoder.

- Local SHA-256: `18d6ae59ba8d893808aae2557ed5e6d1dfb0162ffd5522f191e1f7c9b459da83`
- Checkpoint wrapper: `base_model`
- Original encoder input: XYZ (3 channels)
- Local N2M input: XYZRGB (6 channels)

The local loader copies the pretrained XYZ convolution weights into channels
0--2 and initializes RGB channels 3--5 to zero.  It discards the four
ModelNet classification-head tensors and requires 100% coverage of the local
Point-BERT backbone.  The `.pth` itself is ignored by Git.

