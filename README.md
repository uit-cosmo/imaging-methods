# imaging-methods
Some diverse python codes I have developed and use to analyze imaging data at the boundary of tokamaks. In particular,
I have applied this to APD, Phantom and synthetic blob data. The main method is based on two-dimensional conditional averaging
(2DCA), and other methods work on the output data from 2DCA. The package includes the following files:
- cond_av: Code for two-dimensional conditional averaging
- contours: Blob tracking based on contouring
- data_preprocessing: Method to handle and preprocess experimental data
- discharge: Dataclasses definitions for shots and analysis results
- duration_time_estimation: Method to estimate duration time by fitting power spectral density or auto-correlation function
- fitting: Exponential fits to a profile
- geometry: The last closed flux surface of an EFIT reconstruction
- parameter_estimation: Other estimation methods applied to the outcome of 2DCA.
- show_data: Diverse plotting functions
- utils: Other functions

Top level files have been used to analyze different shots and synthetic data generated with blobmodel.

# Methods

## Two-dimensional conditional averaging (2DCA)

A detailed description of the method is provided in the manuscript prepared to be submitted to PoP: Conditionally averaged blob structures in Alcator C-Mod.
A summary:

A reference pixel is selected, and events are identified where the signal at that pixel exceeds a given threshold.
Within each event, the time of maximum signal amplitude is determined. A window centred at the peak is used to register 
and store the event. Optionally we check that the reference pixel’s signal at the peak is the maximum within a spatial 
radius. Non-compliant events are discarded. Optionally, events are selected to ensure a minimum separation,
prioritized by peak amplitude. Once all events are determined, the events are aligned relative to the peak,
and the average structure is computed across all events. In addition to the average, the conditional 
reproducibility can also be calculated and allows to determine the degree of variability of the events.

## Contouring

Applied on the output of 2DCA. This allows us to estimate the area and center-of-mass velocity without assuming
a specific shape. At each time step of the conditionally averaged event, a contour is drawn at a specified amplitude
level given by a fraction of the maximum amplitude. This contour defines the structure’s boundary at that moment. 
If multiple contours are found at a given frame, the contour enclosing the maximum intensity is chosen. To reduce noise 
and avoid pixel locking, the center of mass signals are filtered with a Hann window. The filtering also helps reducing
the pulsating behaviour obtained when the pulse size is of the order of the spatial resolution of the imaging diagnostic
or smaller. From this, the velocity is computed as the time derivative of the center of mass signal using a centered
differences method. 

<img src="https://github.com/uit-cosmo/phantom/blob/main/figures/example_contour.gif?raw=true" alt="2DCA" style="max-width: 40%;" />

## Ellipse fitting

Applied on the output of 2DCA. 

A two-dimensional Gaussian function with a tilt is fitted to the 2DCA output at time lag zero.

$\varphi(x, y) = A \exp\left( -\left( \frac{x'^2}{\ell_x^2} + \frac{y'^2}{\ell_y^2} \right) \right)$

where $x' = (x - x_\text{ref}) \cos \theta + (y - y_\text{ref}) \sin \theta$ and $y' = -(x - x_\text{ref}) \sin \theta + (y - y_\text{ref}) \cos \theta$. Here, $A$ is the amplitude given by the value of the conditional event at $(x_\text{ref}, y_\text{ref})$, $\ell_x$ and $\ell_y$ are the sizes along the axes, and $\theta$ is the tilt angle.

The fitting process yields $\ell_x$, $\ell_y$ and $\theta$. 

In order to avoid unphysically big, ellongated or tilted blobs, several penalty factors are introduced. The error function to be minimized is

$E(\ell_x, \ell_y, \theta) = \sum_{x, y} (\varphi(\ell_x, \ell_y, \theta; x, y) - data(x, y))^2 + \varphi(\ell_x, \ell_y, \theta; x, y)^2(P_s + P_\theta \theta^2+P_\epsilon(1-\ell_x/\ell_y)^2)$

<img src="https://github.com/uit-cosmo/imaging-methods/blob/main/figures/example_fit.png?raw=true" alt="2DCA" style="max-width: 40%;" />

By convention, lx and ly are defined in such a way that ly is the larger of the two. $\theta \in [0, \pi]$

## Exponential fits

`fit_two_sided(x, y)` fits a two-sided exponential centred at 0 to a profile normalised to 1 at its peak, such as the
cut through a conditional average along R or Z:

$f(x) = c + (1 - c) \exp(-|x|/\lambda), \quad \lambda = s(1 - \sigma) \text{ for } x < 0, \quad \lambda = s\sigma \text{ for } x \ge 0$

The parameters $(s, \sigma, c)$ minimise the sum of squared residuals (L-BFGS-B, with $s > 0$, $0 < \sigma < 1$ and
$|c| \le 10$), starting from the $1/e$ lengths read off the data on either side of the peak. `fit_exp(x, y)` fits
$f(x) = A \exp(x/s)$ as a straight line through $\log y$, leaving out the values that are not positive.

## Last closed flux surface

`calculate_splinted_LCFS(t, efit_time, rbbbs, zbbbs)` takes EFIT's boundary points at the EFIT time nearest $t$, keeps
the outboard leg, $R \ge 86$ cm, and interpolates $R(Z)$ through it with a cubic spline, evaluated at 100 heights from
$Z = -8$ cm to $1$ cm. A pixel's horizontal distance to the separatrix at its own height is then
$R - R_\mathrm{sep}(Z)$. The numbers are C-Mod's, in centimetres, and carry to no other machine.

## Three-point time delay estimation (3TDE)

A detailed description is provided in Phys. Plasmas 32, 042505 (2025)

## Full width half maximum

This method is used to estimate the radial and poloidal characteristic sizes of the average blob. The output of the 2DCA
at time lag 0 is employed. The data points of the average structure at time lag zero are considered for the row (column)
intersecting the reference pixel. The radial (poloidal) size is estimated as the full width half maximum  of the
resuling curve. Due to the low spatial resolution interpolation is employed for the estimation of the full width half maximum.


<img src="https://github.com/uit-cosmo/imaging-methods/blob/main/figures/fwhm_example.png?raw=true" alt="2DCA" style="max-width: 40%;" />

# Overview of blob parameters

| Parameter     | Description                                    |
|---------------|------------------------------------------------|
| vx_c         | contouring                                     |
| vy_c         | contouring                                     |
| area_c       | contouring                                     |
| vx_2dca_tde  | 3TDE on 2DCA output                            |
| vy_2dca_tde  | 3TDE on 2DCA output                            |
| vx_tde       | 3TDE cross-correlation                         |
| vy_tde       | 3TDE cross-correlation                         |
| lx_f         | Ellipse fitting on 2DCA                        |
| ly_f         | Ellipse fitting on 2DCA                        |
| lr           | Full width half maximum on 2DCA                |
| lz           | Full width half maximum on 2DCA                |
| theta_f      | Ellipse fitting on 2DCA                        |
| taud_psd     | PSD fitting for two-sided exponential function |
| lambda_psd   | PSD fitting for two-sided exponential function |
| number_events| Number of events detected with 2DCA            |
