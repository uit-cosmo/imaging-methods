---
# You can also start simply with 'default'
theme: default
background: # some information about your slides (markdown enabled)

title: 2D Conditional Averaging on APD
# apply unocss classes to the current slide
class: text-left
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true

# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
---

# 2D Conditional averaging

Given 2D imaging data, select a reference pixel on which:
- Find all events where the signal is over a **threshold**. 
- For each event:
	- Define the **peak** as the event maximum
	- Register and save the data centered in the peak with a fixed **window size**
	- Optionally, discard the event if any neighbour pixels at a distance lower than check argument have a higher value at the time the peak occurs.
	- Optionally, discard events with overlapping windows with preference for higher amplitude events.
- Return list of events and average over all events

---
transition: slide-left
---
## Arguments
| Name            | Description                          |
|-----------------|--------------------------------------|
| `dataset`       | The 2D imaging data to process      |
| `reference pixel` | Pixel to base events on           |
| `threshold`     | Minimum signal value for events     |
| `window_size`   | Size of data window around peaks    |
| `check_max`     | Distance to check neighboring peaks |
| `single_counting` | Avoid overlapping event windows   |

---
transition: slide-left
---

## Synthetic data

Simulated 5x5 data with 1000 blobs:  
`lx/ly = 1/3`, `theta = -π/4`, `v = 1`, `w = -1`

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/opt_data.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## Result
Averaged output after conditional averaging:

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/opt_out.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## Parameter estimation
Fit to a rotated ellipse to estimate sizes $\ell_x$ and $\ell_y$ and rotation $\theta$

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/2d_ca_fit.png?raw=true){.w-80.mx-auto}

$\ell_x = 1.57$, $\ell_y=0.62$, $\theta = 0.78$

---
transition: slide-left
---

## Results: I mode

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/opt_out_imode.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## Results: L mode

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/opt_out_lmode25.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## L mode events: 1160616018

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/31.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/32.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/33.gif?raw=true){.w-80.mx-auto}
---
transition: slide-left
---

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/34.gif?raw=true){.w-80.mx-auto}
---
transition: slide-left
---

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/35.gif?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## Statistics of events

100ms of data: Events 301, Mean v 439.82, mean w -58.89

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/v_events.png?raw=true){.w-80.mx-auto}

$\sigma = 0.11$

---
transition: slide-left
---

## Velocity u

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/u_events.png?raw=true){.w-80.mx-auto}

$\sigma = -0.05$

---
transition: slide-left
---

# Blob size

Many ways to estimate blob size:

- Exponential fit to poloidal column/ radial row
- Thresholding
- Combination of blob speed and single pixel duration time
- Fit to 2D Gaussian (rotated)

I chose to fit an ellipse because:

- Low spatial resolution might be detrimental to thresholding
- Tilt angle is important in some cases


---
transition: slide-left
---

## 2D Gaussian

Free parameters: $\ell_x, \ell_y, \theta$

Ellipse centre $x_0, y_0$ is taken as the reference pixel

$$
\begin{align}
x_\theta &= (x - x_0) \cos(\theta) + (y - y_0) \sin(\theta) \\
y_\theta &= (y - y_0) \cos(\theta) - (x - x_0) \sin(\theta) \\
\phi(x, y) &= \exp \left( -(x_\theta/\ell_x) ^ 2 - (y_\theta/\ell_y) ^ 2\right) 
\end{align}
$$

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/blob.png?raw=true){.w-80.mx-auto}

---
transition: slide-left
---

## Results

![](https://github.com/uit-cosmo/phantom/blob/main/presentation/event_fits.png?raw=true){.w-80.mx-auto}
