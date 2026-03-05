# Spatial Indexing

survey-sim uses HEALPix (Hierarchical Equal Area isoLatitude Pixelization) to spatially index all survey observations for fast sky-position queries.

## HEALPix Index

The `SpatialIndex` maps each observation to its HEALPix pixel at a configurable NSIDE resolution (default 64, corresponding to ~0.9 deg pixel scale). Queries return all observations in the same pixel and its neighbors.

### Cone Search

For instruments with a well-defined field of view (e.g., Rubin at 1.75 deg radius, ZTF at 3.5 deg), the `query_cone()` method uses the `cdshealpix` Multi-Order Coverage (MOC) algorithm to find all pixels overlapping the FoV circle. This is exact and handles edge cases at the poles.

### Query Flow

```
query(coord, mjd_min, mjd_max)
  ├── HEALPix cone search → candidate pixel set
  ├── Retrieve observation indices from each pixel
  ├── Filter by MJD range [mjd_min, mjd_max]
  └── Return matching observation indices
```

## Performance

The HEALPix index provides \(O(1)\) lookup per pixel. For a typical 10-year survey with ~2M observations:

- Index construction: ~1 second
- Single cone query: ~10 microseconds
- Phase 1 matching for 10K transients: ~100 milliseconds (parallel)

## NSIDE Selection

Higher NSIDE gives finer pixels and fewer false positives per query, but increases memory for the index. The default NSIDE=64 balances well for degree-scale FoVs:

| NSIDE | Pixel scale | Npix | Best for |
|-------|-------------|------|----------|
| 16 | ~3.7 deg | 3,072 | Very wide FoV (Argus) |
| 64 | ~0.9 deg | 49,152 | Standard FoV (Rubin, ZTF) |
| 256 | ~0.2 deg | 786,432 | Narrow FoV instruments |
