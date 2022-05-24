## Simulation Code for  "Measurement by Proxy: On the Accuracy of Online Marketplace Measurements"

The code above provides the infrastructure to simulate marketplaces as defined in our paper's model. Following,
we provide an overview of what's needed to use this code.

### Dependencies

No extraneous libraries are required beyond `numpy`, `pandas`, and `datetime`. To use the Jolly Seber abundance
estimator we need to have the [RMark package](https://cran.r-project.org/web/packages/RMark/index.html).
However, the simulations can still be conducted without.

### Configuration

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

