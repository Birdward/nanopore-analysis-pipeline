# High-Throughput Nanopore Event Analysis Pipeline

## Project Overview

This repository contains version 1 of the complete Python pipeline developed for my MSci thesis project at Imperial College London. The goal was to automate the analysis of nanopore data to identify and characterise translocation events. The pipeline implements a full workflow: data loading, preprocessing, event detection, classification based on event morphology, and detailed parametric analysis using both the ADEPT and CUSUM algorithms. This project was later incorporated into the final nanopore analysis program.

## Project Structure

The project is structured into modular components to ensure clarity and maintainability:

* `main.py`: The main execution script that orchestrates the entire pipeline from data loading to saving results. It's designed to be configurable for different datasets and parameters.
* `Data_Load_Preprocessing.py`: Handles loading raw .abf files, signal filtering, downsampling, and advanced baseline correction (e.g. Asymmetric Least Squares).
* `Event_Detection.py`: Implements the initial event detection logic using Gaussian fitting to characterise noise and determine signal boundaries.
* `Event_classification.py`: A rule-based classifier that categorises events as either short, pulse-like (ADEPT) or long, multi-level (CUSUM) based on duration, SNR, and steady-state analysis.
* `ADEPT.py`: Contains the implementation of the ADEPT model for analysing short translocation events via non-linear curve fitting.
* `CUSUM.py`: Implements the CUSUM algorithm to detect discrete state changes and sub-peaks within complex, multi-level events.
* `Visualisation_Output.py`: Manages the output, including generating plots for quality control and saving all extracted event parameters to a structured .csv file for downstream analysis.

## Key Features & Technical Highlights

* End-to-End Automation: The pipeline fully automates the analysis of raw signal data into a final, quantified results table.
* Algorithmic Depth: Implements both established (ADEPT) and statistical change-point (CUSUM) algorithms to handle diverse event types, showcasing the ability to translate complex scientific methods into reliable code.
* Production-Ready Output: The save_event_params_csv function demonstrates a focus on producing clean, well-structured, and machine-readable data suitable for large-scale studies and machine learning models.
* Modularity & Reusability: Each step of the analysis is encapsulated in its own module and functions, following good software engineering principles.
