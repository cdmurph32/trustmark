// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

//! # Trustmark
//!
//! An implementation of TrustMark watermarking for the Content Authenticity Initiative (CAI) in
//! Rust, as described in:
//!
//! ---
//!
//! **TrustMark - Universal Watermarking for Arbitrary Resolution Images**
//!
//! <https://arxiv.org/abs/2311.18297>
//!
//! [Tu Bui]<sup>1</sup>, [Shruti Agarwal]<sup>2</sup>, [John Collomosse]<sup>1,2</sup>
//!
//! <sup>1</sup>DECaDE Centre for the Decentralized Digital Economy, University of Surrey, UK.\
//! <sup>2</sup>Adobe Research, San Jose CA.
//!
//! ---
//!
//! This is a re-implementation of the [trustmark] Python library.
//!
//! [Tu Bui]: https://www.surrey.ac.uk/people/tu-bui
//! [Shruti Agarwal]: https://research.adobe.com/person/shruti-agarwal/
//! [John Collomosse]: https://www.collomosse.com/
//! [trustmark]: https://pypi.org/project/trustmark/
//!
//! ## Example
//!
//! ```rust
//! use trustmark::{Trustmark, Version, Variant};
//!
//! # fn main() {
//! let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
//! let input = image::open("../images/ghost.png").unwrap();
//! let output = tm.encode("0010101".to_owned(), input, 0.95);
//! # }
//! ```
use std::path::Path;

use image::{DynamicImage, GenericImageView as _};
#[cfg(not(target_arch = "wasm32"))]
use ort::{GraphOptimizationLevel, Session};

#[cfg(target_arch = "wasm32")]
use wonnx::Session;

use self::{bits::Bits, image_processing::ModelImage};

mod bits;
mod image_processing;
mod model;

/// A loaded Trustmark model.
pub struct Trustmark {
    encoder: Option<Session>,
    decoder: Option<Session>,
    version: Version,
    variant: Variant,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("watermark is corrupt or missing")]
    CorruptWatermark,
    #[error("onnx error: {0}")]
    #[cfg(not(target_arch = "wasm32"))]
    Ort(#[from] ort::Error),
    #[cfg(target_arch = "wasm32")]
    #[error("wonnx session error: {0}")]
    Wonnx(#[from] wonnx::SessionError),
    #[cfg(target_arch = "wasm32")]
    #[error("invalid input")]
    InvalidInput,
    #[cfg(target_arch = "wasm32")]
    #[error("invalid output")]
    InvalidOutput,
    #[cfg(target_arch = "wasm32")]
    #[error("invalid shape: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("image processing error: {0}")]
    ImageProcessing(#[from] image_processing::Error),
    #[error("bits processing error: {0}")]
    Bits(bits::Error),
    #[error("invalid model variant")]
    InvalidModelVariant,
    #[error("model not found: {0}")]
    ModelNotFound(String),
}

impl From<bits::Error> for Error {
    fn from(value: bits::Error) -> Self {
        match value {
            bits::Error::CorruptWatermark => Error::CorruptWatermark,
            err => Error::Bits(err),
        }
    }
}

pub use bits::Version;
pub use model::Variant;

impl Trustmark {
    /// Load a Trustmark model.
    pub fn new<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        version: Version,
    ) -> Result<Self, Error> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let encoder = Some(
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(8)?
                    .commit_from_file(models.as_ref().join(variant.encoder_filename()))?,
            );
            let decoder = Some(
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(8)?
                    .commit_from_file(models.as_ref().join(variant.decoder_filename()))?,
            );
            Ok(Self {
                encoder,
                decoder,
                version,
                variant,
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let encoder = Some(pollster::block_on(Session::from_path(
                models.as_ref().join(variant.decoder_filename()),
            ))?);
            let decoder = Some(pollster::block_on(Session::from_path(
                models.as_ref().join(variant.decoder_filename()),
            ))?);
            Ok(Self {
                encoder,
                decoder,
                version,
                variant,
            })
        }
    }

    /// Load a single model from memory
    pub fn new_from_bytes(
        model: &[u8],
        variant: Variant,
        version: Version,
        is_encoder: bool, // TODO: make an enum
    ) -> Result<Self, Error> {
        if is_encoder {
            let encoder: Option<Session>;
            #[cfg(not(target_arch = "wasm32"))]
            {
                encoder = Some(
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .with_intra_threads(8)?
                        .commit_from_memory(model)?,
                );
            }
            #[cfg(target_arch = "wasm32")]
            {
                encoder = Some(pollster::block_on(Session::from_bytes(model))?);
            }
            Ok(Self {
                encoder,
                decoder: None,
                version,
                variant,
            })
        } else {
            let decoder: Option<Session>;
            #[cfg(not(target_arch = "wasm32"))]
            {
                decoder = Some(
                    Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .with_intra_threads(8)?
                        .commit_from_memory(model)?,
                );
            }
            #[cfg(target_arch = "wasm32")]
            {
                decoder = Some(pollster::block_on(Session::from_bytes(model))?);
                Ok(Self {
                    encoder: None,
                    decoder,
                    version,
                    variant,
                })
            }
        }
    }

    /// Encode a watermark into an image.
    ///
    /// `watermark` is a bitstring encoding the watermark identifier to encode. `img` is the image
    /// which will be watermarked. `strength` is a number between 0 and 1 indicating how strong the
    /// resulting watermark should be. 0.95 is a normal strength.
    pub fn encode(
        &self,
        watermark: String,
        img: DynamicImage,
        strength: f32,
    ) -> Result<DynamicImage, Error> {
        let (original_width, original_height) = img.dimensions();
        let aspect_ratio = original_width as f32 / original_height as f32;

        // the image is always encoded with size 256x256
        let encode_size = 256;

        let encoder = self
            .encoder
            .as_ref()
            .ok_or(Error::ModelNotFound("No encoder model session".to_string()))?;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let input_img: ort::Value<ort::TensorValueType<f32>> =
                ModelImage(encode_size, self.variant, img.clone()).try_into()?;
            let bits: ort::Value<ort::TensorValueType<f32>> =
                Bits::apply_error_correction_and_schema(watermark, self.version)?.into();
            let outputs = encoder.run(ort::inputs![
                "onnx::Concat_0" => input_img,
                "onnx::Gemm_1" => bits,
            ]?)?;
            let output_img = outputs["image"].try_extract_tensor::<f32>()?.to_owned();

            // Need to calculate and apply the residual.
            let input_img: ort::Value<ort::TensorValueType<f32>> =
                ModelImage(encode_size, self.variant, img.clone()).try_into()?;
            let residual = (self.variant.strength_multiplier() * strength)
                * (output_img - input_img.try_extract_tensor::<f32>()?);

            // Residual should be small perturbations.
            let mut residual = residual.clamp(-0.2, 0.2);
            if (self.variant == Variant::Q && !(0.5..=2.0).contains(&aspect_ratio))
                || self.variant == Variant::P
            {
                residual = image_processing::remove_boundary_artifact(
                    residual,
                    (original_width as usize, original_height as usize),
                    self.variant,
                );
            }

            let ModelImage(_, _, residual) = (encode_size, self.variant, residual).try_into()?;

            Ok(image_processing::apply_residual(img, residual))
        }
        #[cfg(target_arch = "wasm32")]
        {
            use std::borrow::Cow;
            use std::collections::HashMap;
            use wonnx::utils::InputTensor;

            // Prepare input tensors (convert to f32 arrays as needed)
            let input_img: ndarray::ArrayD<f32> =
                ModelImage(encode_size, self.variant, img.clone()).try_into()?;
            let bits: ndarray::ArrayD<f32> =
                Bits::apply_error_correction_and_schema(watermark, self.version)?.into();

            // Prepare wonnx inputs using InputTensor
            let mut inputs = HashMap::new();
            inputs.insert(
                "onnx::Concat_0".to_string(),
                InputTensor::F32(Cow::Borrowed(
                    input_img.as_slice().ok_or(Error::InvalidInput)?,
                )),
            );
            inputs.insert(
                "onnx::Gemm_1".to_string(),
                InputTensor::F32(Cow::Borrowed(bits.as_slice().ok_or(Error::InvalidInput)?)),
            );

            // Run wonnx session
            let outputs = pollster::block_on(encoder.run(&inputs)).map_err(Error::from)?;
            let output_img = outputs
                .get("image")
                .ok_or(Error::ImageProcessing(image_processing::Error::Image))?;

            // Convert OutputTensor to Vec<f32>
            let output_img = match output_img {
                wonnx::utils::OutputTensor::F32(v) => {
                    ndarray::ArrayD::from_shape_vec(input_img.raw_dim(), v.clone())?
                }
                _ => return Err(Error::InvalidOutput),
            };

            // Calculate and apply the residual
            let input_img: ndarray::ArrayD<f32> =
                ModelImage(encode_size, self.variant, img.clone()).try_into()?;
            let residual =
                (self.variant.strength_multiplier() * strength) * (&output_img - &input_img);

            let mut residual = residual.mapv(|x| x.clamp(-0.2, 0.2));
            if (self.variant == Variant::Q && !(0.5..=2.0).contains(&aspect_ratio))
                || self.variant == Variant::P
            {
                residual = image_processing::remove_boundary_artifact(
                    residual,
                    (original_width as usize, original_height as usize),
                    self.variant,
                );
            }
            let residual_img = image_processing::array_to_image(residual)?;
            Ok(image_processing::apply_residual(img, residual_img))
        }
    }

    /// Decode a watermark from an image.
    pub fn decode(&self, img: DynamicImage) -> Result<String, Error> {
        // P variant has a smaller decode size
        let decode_size = if self.variant == Variant::P { 224 } else { 256 };

        let decoder = self
            .decoder
            .as_ref()
            .ok_or(Error::ModelNotFound("No decoder model session".to_string()))?;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let img: ort::Value<ort::TensorValueType<f32>> =
                ModelImage(decode_size, self.variant, img).try_into()?;
            let outputs = decoder.run(ort::inputs![
                "image" => img,
            ]?)?;
            let watermark = outputs["output"].try_extract_tensor::<f32>()?.to_owned();
            let watermark: Bits = watermark.try_into()?;
            Ok(watermark.get_data())
        }
        #[cfg(target_arch = "wasm32")]
        {
            use std::borrow::Cow;
            use std::collections::HashMap;
            use wonnx::utils::InputTensor;
            let input_img: ndarray::ArrayD<f32> =
                ModelImage(decode_size, self.variant, img).try_into()?;

            let mut inputs = HashMap::new();
            inputs.insert(
                "image".to_string(),
                InputTensor::F32(Cow::Borrowed(
                    input_img.as_slice().ok_or(Error::InvalidInput)?,
                )),
            );

            let outputs = pollster::block_on(decoder.run(&inputs)).map_err(Error::from)?;
            let output = outputs.get("output").ok_or(Error::InvalidOutput)?;

            let bits = match output {
                wonnx::utils::OutputTensor::F32(v) => {
                    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 100]), v.clone())
                        .map_err(|_| Error::InvalidOutput)?;
                    Bits::try_from(arr)?
                }
                _ => return Err(Error::InvalidOutput),
            };
            Ok(bits.get_data())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_models() {
        Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
    }

    fn roundtrip(path: impl AsRef<Path>) {
        let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let input = image::open(path.as_ref()).unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = tm.encode(watermark.clone(), input, 0.95).unwrap();
        encoded.to_rgba8().save("./test.png").unwrap();
        let input = image::open("./test.png").unwrap();
        let decoded = tm.decode(input).unwrap();
        assert_eq!(watermark, decoded);
    }

    #[test]
    fn roundtrip_ghost() {
        roundtrip("../images/ghost.png");
    }

    #[test]
    fn roundtrip_ufo() {
        roundtrip("../images/ufo_240.jpg");
    }
}
