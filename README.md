# TI AWR1642 & MR60BHA2 Stable Vital Signs Node for ROS 2

This repository contains a ROS 2 node that interfaces with the Texas Instruments AWR1642 mmWave Radar to extract point cloud data and perform real-time estimation of a subject's Heart Rate (BPM) using phase-based signal processing.

---

## Overview

The **StableAWR1642Node** is designed to provide a steady and reliable heart rate measurement by analyzing the micro-displacements of a person's chest.

Unlike raw radar data, which can be noisy and erratic, this node implements several signal processing techniques to ensure the output BPM is smooth and accurate.

---

## Key Features

- **Automated Configuration**  
  Sends the required CLI configuration strings to the radar hardware upon startup.

- **Phase Unwrapping**  
  Handles $2\pi$ phase discontinuities to maintain signal integrity over time.

- **Advanced Filtering**  
  Uses a 4th-order Butterworth bandpass filter ($0.8 \text{Hz}$ – $2.5 \text{Hz}$) to isolate cardiac vibrations.

- **Stability Enhancements**  
  Implements Hanning windowing to reduce spectral leakage and an exponential smoothing filter for the final BPM output.

