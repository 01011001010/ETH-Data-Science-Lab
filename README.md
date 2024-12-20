# Few-Shot Anomaly Detection in Aero-Engine Blade Inspection

This project evaluates state-of-the-art **few-shot anomaly classification** methods on the **AeBAD dataset**, a real-world aero-engine blade inspection dataset. We specifically implement and test **WinCLIP** and **WinCLIP+**, leveraging **zero-shot** and **few-shot learning** paradigms to improve anomaly classification and localization. The objective is to demonstrate the robustness of few-shot methods in low-data regimes and ensure fair comparison with existing benchmarks.

## Project Structure

```
..
├── WinCLIP                 # WinCLIP implementation for anomaly detection
├── MMR_fewer_shot          # MMR Benchmark (Reduced-Data Evaluation)
├── report.pdf              # The main report of this project
├── poster.pdf              # Poster from our poster presentation
├── LICENCE                 # MIT Licence
├── README.md               # This README
```

## Key Components

1. **WinCLIP (Zero-Shot)**  
WinCLIP leverages a pre-trained vision-language model for anomaly detection without any fine-tuning. It demonstrates strong generalization capabilities but lacks pixel-level alignment in certain cases.

2. **WinCLIP+ (Few-Shot Learning)**  
   Incorporating a small number of normal samples (1–4 shots) significantly enhances WinCLIP's performance. This reduces false positives and improves anomaly localization, especially in scenarios involving domain shifts (e.g., illumination or background changes).

3. **MMR Benchmark (Reduced-Data Evaluation)**  
   MMR, a reconstruction-based method, is evaluated on smaller subsets of AeBAD. This provides a balanced benchmark for comparison with WinCLIP+.

4. **Benchmark Comparisons**  
   The project evaluates the Mean AUROC (%) of WinCLIP+ against full-data benchmarks like PatchCore, ReverseDistillation, and other state-of-the-art methods.

## Results & Discussion

The experiments evaluated the performance of WinCLIP and WinCLIP+ across various settings. Key observations include:

- **Zero-Shot Learning**: WinCLIP achieves reasonable performance without requiring labeled data but is limited in fine-grained anomaly segmentation.
- **Few-Shot Learning**: WinCLIP+ achieves significant performance improvements, demonstrating the benefits of incorporating minimal training data.
- **Fair Comparison**:  WinCLIP+ was extended to 4 shots for direct comparison with benchmarks trained on full datasets. MMR was evaluated on a reduced dataset to ensure balanced baseline evaluation.



<table>
    <tr>
        <td><b>Source</b></td>
        <td><b>Method</b></td>
        <td><b>Same</b></td>
        <td><b>Background</b></td>
        <td><b>Illumination</b></td>
        <td><b>View</b></td>
        <td><b>Mean</b></td>
    </tr>
    <tr>
        <td rowspan="6"><a href="https://doi.org/10.48550/arXiv.2304.02216">Zhang et al.</a></td>
        <td>PatchCore</td>
        <td>75.2 ± 0.3</td>
        <td>74.1 ± 0.3</td>
        <td>74.6 ± 0.4</td>
        <td>60.1 ± 0.4</td>
        <td>71.0</td>
    </tr>
    <tr>
        <td>ReverseDistillation</td>
        <td>82.4 ± 0.6</td>
        <td>84.3 ± 0.9</td>
        <td>85.5 ± 0.9</td>
        <td>71.9 ± 0.8</td>
        <td>81.0</td>
    </tr>
    <tr>
        <td>DRAEM</td>
        <td>64.0 ± 0.4</td>
        <td>62.1 ± 6.1</td>
        <td>61.6 ± 2.7</td>
        <td>62.3 ± 0.9</td>
        <td>62.5</td>
    </tr>
    <tr>
        <td>NSA</td>
        <td>66.5 ± 1.4</td>
        <td>48.8 ± 3.5</td>
        <td>55.5 ± 3.2</td>
        <td>55.9 ± 1.1</td>
        <td>56.7</td>
    </tr>
    <tr>
        <td>RIAD</td>
        <td>38.6 ± 0.6</td>
        <td>41.6 ± 1.3</td>
        <td>46.8 ± 0.8</td>
        <td>33.0 ± 0.6</td>
        <td>40.0</td>
    </tr>
    <tr>
        <td>InTra</td>
        <td>39.8 ± 0.8</td>
        <td>46.1 ± 0.5</td>
        <td>44.7 ± 0.3</td>
        <td>46.3 ± 1.5</td>
        <td>44.2</td>
    </tr>
    <tr>
        <td rowspan="4">Our work</td>
        <td>MMR (Recreated Benchmark)</td>
        <td><b>85.6 ± 0.5</b></td>
        <td><b>84.4 ± 0.7</b></td>
        <td><b>88.8 ± 0.5</b></td>
        <td>79.9 ± 0.6</td>
        <td><b>84.7</b></td>
    </tr>
    <tr>
        <td>WinCLIP+ (0-Shot)</td>
        <td>80.3 ± 0.2</td>
        <td>82.9 ± 0.5</td>
        <td>67.0 ± 0.3</td>
        <td>82.0 ± 0.3</td>
        <td>78.0</td>
    </tr>
    <tr>
        <td>WinCLIP+ (1-Shot)</td>
        <td>80.7 ± 0.5</td>
        <td>83.1 ± 0.5</td>
        <td>67.4 ± 0.6</td>
        <td><b>82.1 ± 0.4</b></td>
        <td>78.3</td>
    </tr>
    <tr>
        <td>WinCLIP+ (4-Shot)</td>
        <td>80.9 ± 0.2</td>
        <td>83.7 ± 0.4</td>
        <td>67.7 ± 0.4</td>
        <td>81.9 ± 0.3</td>
        <td>78.6</td>
    </tr>
</table>

### Acknowledgements

This work was completed as part of the ETH Zurich Data Science Lab. Special thanks to the ETH AI Center and IBM Research Europe for their support and collaboration.

### License

This project is released under an [MIT License](LICENCE).
