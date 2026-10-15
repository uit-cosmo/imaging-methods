# imaging-methods
Some diverse python codes I have developed and use to analyze imaging data at the boundary of tokamaks. In particular,
I have applied this to APD, Phantom and synthetic blob data. The main method is based on two-dimensional conditional averaging
(2DCA), and other methods work on the output data from 2DCA. The package includes the following files:
- cond_av: Code for two-dimensional conditional averaging
- contours: Blob tracking based on contouring
- data_preprocessing: Method to handle and preprocess experimental data
- dead_pixels: Dead APD pixels estimated from the raw record
- discharge: Dataclasses definitions for shots and analysis results
- duration_time_estimation: Method to estimate duration time by fitting power spectral density or auto-correlation function
- parameter_estimation: Other estimation methods applied to the outcome of 2DCA.
- puff: The gas-puff window of a raw APD record
- show_data: Diverse plotting functions
- utils: Other functions

Top level files have been used to analyze different shots and synthetic data generated with blobmodel.

# Methods

## Dead pixels

A pixel is live when it sees the plasma, and the evidence is in its fluctuations: their spectrum, and whether they
follow a live neighbour's. Its PDF alone cannot tell: a dim pixel inside the separatrix has a narrow, near-Gaussian PDF
just as dead ones do, and a dead channel can be skewed by rare spikes or by telegraph noise. `dead_pixels.estimate`
judges every pixel of a raw record over the window it is given:

1. **Red spectrum** (`red_ratio`). With $S(f)$ the Welch power spectral density of the pixel (segments of 4096 samples,
`NPERSEG`), the ratio $\rho = \mathrm{median}_{1-20\ \mathrm{kHz}}\ S\ /\ \mathrm{median}_{300-900\ \mathrm{kHz}}\ S$
compares the blob band (`RED_BAND`) with a band where only digitizer noise is left (`NOISE_BAND`); the medians ignore
the narrow pickup lines in that band. Noise is white, $\rho \approx 1$, and plasma fluctuations are red: a pixel with
$\rho > 150$ (`RED`) is live.
2. **Following a live neighbour** (`band_passed`, `neighbour_gains`). With $b_p$ the pixel's signal band-passed to 2-20
kHz (`BLOB_BAND`: block averages over $1/(2 f_\mathrm{high})$, less their running mean over $1/f_\mathrm{low}$), the
regression gain of pixel $p$ on its neighbour $q$ is $g_{pq} = \overline{b_p b_q}/\sigma_q^2 = r_{pq}\sigma_p/\sigma_q$.
A pixel that fails step 1 is live when its gain on one of its eight neighbours that is already live exceeds 0.1
(`GAIN`), repeated until no pixel joins. This keeps dim pixels whose own fluctuations are weak (on C-Mod, mostly in the
innermost columns). It is the gain and not the correlation, because dead channels share pickup with one another and
correlate strongly while carrying nothing, and crosstalk carries only a few percent of a neighbour's signal.
3. Every other pixel is dead, and so is a pixel with no samples (non-finite or constant).

The result holds the mask, each pixel's evidence (`NO_DATA`, `NO_EVIDENCE`, `FOLLOWS_NEIGHBOUR` or `RED_SPECTRUM`), its
$\rho$ and its largest gain on a live neighbour. `consolidate` gives a run day's mask from its shots' masks: dead where
dead in at least a third of them (`DAY_FRACTION`), since the hardware does not change within a day.

## The gas-puff window

On many shots the record starts before the gas puff, while the live pixels still read the digitizer's dark level, and a
window that includes that dark stretch gives bimodal PDFs and a record that does not start stationary.
`puff.puff_window(record, (t_start, t_end))` cuts the discharge window to the puff, found on the whole record
(`find_puff` does the same on a 1-D array mean). The dark level is read off the start of what it is given, so it is
never given a record already cut to the discharge window, whose start may be after the puff:

1. The signal $\bar{s}(t)$ is the mean over the pixels finite over the whole record, smoothed by a 1 ms running mean
(`SMOOTH`).
2. The baseline $b$ is its median over the first 5 ms of the record (`BASELINE_SPAN`), and the level $\ell$ its median
over the discharge window. The polarity is read off the data: the plasma lies on the side of $b$ where the window's 5th
or 95th percentile lies further (on C-Mod more light reads higher, from a dark level of about -1.11 V). Where the median
is not on that side, or is less than a quarter of the way from $b$ to that excursion (`CLEAR_RISE`), as when most of the
window is dark after a short puff and the median is the dark level, the window's 90th percentile (`FALLBACK_QUANTILE`;
the 10th, where light reads lower) is the level, provided that at this level the rise is clear and the start of the
record flat (step 4). A record that does not start dark fails the second test at any level, so it keeps the median's
verdict: its window is not read from a decay. The threshold is $\theta = (b + \ell)/2$.
3. The puff runs from the first point at which $\bar{s}$ crosses $\theta$ and stays across it for at least 5 ms
(`MIN_ON`), to the last point of the last such run. The analysis window is its overlap with the discharge window. A cut
shorter than the smoothing is not made. A dip below $\theta$ inside the window, as between two puffs, is reported and
not cut; a dip shorter than 1 ms (`MIN_DIP`) is noise at the threshold, and not reported.
4. The discharge window is kept whole, and the result says why, when the record does not start dark (its first 5 ms vary
by more than 4% of the rise $|\ell - b|$, `FLAT_START`, or $\bar{s}$ is across $\theta$ from its first point), when
there is no clear rise, when $\bar{s}$ never stays across $\theta$ for 5 ms, or when the puff overlaps the window by
less than 20 ms (`MIN_SPAN`).

The result, a `PuffWindow`, carries the smoothed signal, its levels and the rule in words (`rule`), which preprocessing
stores with the file.

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
