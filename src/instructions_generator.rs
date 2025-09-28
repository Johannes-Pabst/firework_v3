use crate::{colors::wavelength_to_stimul, instructions_helper::{Curve, CurvePoint, Helper, ParticleInstructions, Spawner}, spectra::spectrum_file_to_rgb};

pub struct Rocket {}
impl Rocket {
    pub fn new() -> Self {
        Rocket {}
    }
    pub fn generate_instructions(
        &self,
        inst_h: &mut Helper<ParticleInstructions>,
        spwn_h: &mut Helper<Spawner>,
        c_buf: &mut Vec<CurvePoint>,
    ) -> u32 {
        let stars=inst_h.save(
            ParticleInstructions::new(
                c_buf,
                1.0,
                Curve::new(
                    vec![
                        CurvePoint::new(-120.0, 0.2),
                        CurvePoint::new(-60.0, 0.02),
                        CurvePoint::new(0.0, 0.0),
                    ],
                    1,
                ) * 50.0
                    * spectrum_file_to_rgb("spectrums_nist_gov/Sr.html"),
            ),
        );
        let stars_sp=spwn_h.save(
            Spawner::new(
                c_buf,
                Curve::new(
                    vec![CurvePoint::new(-3.0, 0.0), CurvePoint::new(-3.0, 50.0)],
                    1,
                ),
                Curve::new(
                    vec![CurvePoint::new(90.0, 1.0), CurvePoint::new(120.0, 0.0)],
                    2,
                ),
                0.0,
                10.0,
                10.0,
                stars,
            ),
        );
        let exhaust=inst_h.save(
            ParticleInstructions::new(
                c_buf,
                5.0,
                Curve::new(
                    vec![
                        CurvePoint::new(-60.0, 0.2),
                        CurvePoint::new(-30.0, 0.02),
                        CurvePoint::new(0.0, 0.0),
                    ],
                    1,
                ) * 0.2
                    * wavelength_to_stimul(600.0),
            ),
        );
        let exhaust_sp=spwn_h.save(
            Spawner::new(
                c_buf,
                Curve::new(
                    vec![
                        CurvePoint::new(0.0, 40.0),
                        CurvePoint::new(30.0, 40.0),
                        CurvePoint::new(50.0, 4.0),
                        CurvePoint::new(60.0, 0.4),
                        CurvePoint::new(70.0, 0.0),
                    ],
                    0,
                ) * 100.0,
                Curve::new(
                    vec![
                        CurvePoint {
                            _t: 30.0,
                            _v: 1.0,
                            _buffer: [0.0, 0.0],
                        },
                        CurvePoint {
                            _t: 40.0,
                            _v: 0.1,
                            _buffer: [0.0, 0.0],
                        },
                        CurvePoint {
                            _t: 50.0,
                            _v: 0.01,
                            _buffer: [0.0, 0.0],
                        },
                        CurvePoint {
                            _t: 60.0,
                            _v: 0.0,
                            _buffer: [0.0, 0.0],
                        },
                    ],
                    2,
                ),
                1.5,
                0.5,
                1.0,
                exhaust,
            ),
        );
        let rocket=inst_h.save(
            ParticleInstructions::new(
                c_buf,
                0.1,
                Curve::fr_cst(0.0) * wavelength_to_stimul(550.0),
            )
            .with_v_thruster(
                c_buf,
                Curve::new(
                    vec![
                        CurvePoint::new(0.0, 30.0),
                        CurvePoint::new(30.0, 30.0),
                        CurvePoint::new(60.0, 0.0),
                    ],
                    0,
                ),
            )
            .with_v_thruster_spawner(exhaust_sp)
            .with_passive_spawner(stars_sp),
        );
        spwn_h.save(
            Spawner::new(
                c_buf,
                1.0 / 5.0,
                Curve::new(
                    vec![
                        CurvePoint {
                            _t: 80.0,
                            _v: 1.0,
                            _buffer: [0.0, 0.0],
                        },
                        CurvePoint {
                            _t: 90.0,
                            _v: 0.0,
                            _buffer: [0.0, 0.0],
                        },
                    ],
                    2,
                ),
                1.5,
                0.5,
                1.5,
                rocket,
            ),
        )
    }
}
