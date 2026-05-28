# material_reconstruction
For material reconstruction from CT image

Weights
Weight file is: https://github.com/daiyufujiwara/material_reconstruction/releases/tag/v1.0.0

Usage
Unet.py predicts material distribution from the CT image (FBP_120kV_AF.raw) with the model architecture and corresponding weight file.

Input Data Specifications
To perform material reconstruction using clinical data or custom CT images, the input data must satisfy the following specifications:
- Image Size: 256 x 256 (can be resized from 512 x 512)
- Data Type: float32
- File Format: .raw

Data Preprocessing (HU to Linear Attenuation Coefficient)
The input images for this model must be calibrated in linear attenuation coefficients ($\mu$), NOT Hounsfield Units (HU). 

Therefore, if you are using clinical CT data (which typically uses HU), you must convert the values before feeding them into the model.

The detailed conversion formula from HU to the linear attenuation coefficient ($\mu$) is as follows:

$$\mu = \mu_{\text{water}} \times \left(1 + \frac{\text{HU}}{1000}\right)$$

Where:
$\mu$ is the linear attenuation coefficient to be input into the model.

$\text{HU}$ is the Hounsfield Unit value from the clinical CT data.

$\mu_{\text{water}}$ is the linear attenuation coefficient of water at the corresponding X-ray tube voltage (e.g., 0.233150 for 120 kV in our case).
