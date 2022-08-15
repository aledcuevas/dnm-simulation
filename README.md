## Simulation Code for  "Measurement by Proxy: On the Accuracy of Online Marketplace Measurements"

The code above provides the infrastructure to simulate marketplaces as defined in our paper's model. Following,
we provide an overview of what's needed to use this code.

For the public data used in the paper, see: https://arima.cylab.cmu.edu/markets/index.php

A anonymized version can also be requested at: https://www.impactcybertrust.org/dataset_view?idDataset=1498

A non-anonymized version can be requested at: https://www.impactcybertrust.org/dataset_view?idDataset=1499

### Artifact Evaluation Roadmap

Following we provide a roadmap that aims to provide the reader with an understanding of: 1) the role 
of the code in our paper, 2) what the expected inputs and outputs are, and 3) how to operate the code
to obtain and evaluate the outputs.

#### Overview and Goal
The above code aims to simulate the evolution of an artificial online marketplace. To do that, it will take as input 
various parameters that probabilistically describe the creation and interaction of objects in the marketplace (e.g., what's 
the probability of a new item appearing in day X)?

For evaluation, we provide a set of dummy parameters that would allow the code to run. Note that, as described in our paper,
we are not able to release the empirical distributions that we use for the paper. This is because of the agreement with our law
enforcement partners. For more details, refer to the paper.

#### Dependencies

We used `Python 3.8.10` to test the code. No extraneous libraries are required beyond `numpy`, `pandas`, and `datetime`.
To use the Jolly Seber abundance estimator, the [RMark package](https://cran.r-project.org/web/packages/RMark/index.html) is needed.
However, this is not required to conduct the simulations.

#### Running the Simulation & Expected Behavior
To run, from the directory in your terminal do:
```python
python3 ./main.py
```
The code will create the following file structure:
```commandline
    .
    ├── main.py                     
    └── results/                    
            └── <%m.%D-%H:%M_s>
                    ├── pr_parameters.json
                    ├── scraper_parameters.json
                    ├── sim_parameters.json
                    ├── simulation_config_summary.txt
                    └── serialized/
                            ├── states/
                            ├── <%m.%D-%H:%M_s>
                            └── <scraper>/
                                    └── states
```


### Configuration Advice for Future Researchers

There's no one way to simulate a marketplace. At its core, what we are doing is defining various probability 
distributions for the various objects and events that occur in a marketplace. We can use well-defined distributions
or we can use empirical ones. We can decide to have vendors appear at a constant or exponential rate. We can decide
for there to be no deletions or that everything be deleted after a week, and so on. We provide a template
to simulate Hansa-like markets.

More specifically, you will find a standard configuration ready to be utilized. As of now, what needs to be supplied are:

- `EMP_LISTING_P_VENDOR`: we need a distributions of values for the number of listings that a vendor could have.
  - This is used in `sample_listings_per_vendor()` to create a weight or 'propensity' of how likely that vendor is to be assigned a listing
- `EMP_FB_PER_LISTING`: same as above but for distribution of feedbacks per listing.
    - This is used to also create a 'propensity' in `sample_listing_popularity()`

**Note**: We have experimented with also defining a sampling space for prices using the above procedure which can
be tested with `EMP_LISTING_PRICES`. However, we've observed that a meaningful interpretation of artificial revenue
is trickier given that item purchasing behavior has several more latent factors that impact whether it is bought or not,
thus defining a probability space is harder than just using frequency as a weight.

Other parameters that ought to be tweaked for the simulation are encased in the following variables:
`SIMULATION_PARAMETERS`, `SCRAPING_PARAMETERS`, and `PR_PARAMETERS`. 

