use survey_sim::spatial::SpatialIndex;

#[test]
fn test_healpix_index_roundtrip() {
    // Create 100 random coordinates and verify they all can be queried back.
    let coords: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let ra = (i as f64 * 3.6) % 360.0;
            let dec = -90.0 + (i as f64 * 1.8);
            (ra, dec.clamp(-89.9, 89.9))
        })
        .collect();

    let index = SpatialIndex::new(&coords, 64);

    for (i, &(ra, dec)) in coords.iter().enumerate() {
        let results = index.query(ra, dec);
        assert!(
            results.contains(&i),
            "Coordinate {} ({}, {}) not found in its own pixel",
            i,
            ra,
            dec
        );
    }
}

#[test]
fn test_healpix_neighbor_query() {
    let coords = vec![(45.0, 30.0)];
    let index = SpatialIndex::new(&coords, 64);

    // Query with neighbors should also return the point.
    let results = index.query_with_neighbors(45.0, 30.0);
    assert!(results.contains(&0));
}
