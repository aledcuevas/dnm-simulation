'''
Code to simulate a marketplace for the paper 'Measurement By Proxy' USENIX '22
by Alejandro Cuevas, May 2022
'''


import os
import multiprocessing as mp
import subprocess
import json
import copy
import time
from datetime import datetime

import math
import random
import numpy as np
import pandas as pd

import statistics, utility

pd.set_option('display.float_format', lambda x: '%.5f' % x)

SMA_DICTIONARY =                None
BASE_SIMULATION_RESULT_FOLDER = './results/'

EMP_LISTING_PRICES =    None
EMP_LISTINGS_P_VENDOR = None
EMP_FB_PER_LISTING =    None

MARKETPLACE_EVENTS = {"NEW_ITEM": 0,
                      "REVIEW": 0,
                      "SET_HOLDING_PRICE": 0,
                      "UNSET_HOLDING_PRICE": 0,
                      "DELETE_ITEM": 0,
                      "HIDE_ITEM": 0,
                      "UNHIDE_ITEM": 0,
                      'NEW_VENDOR': 0}

PR_PARAMETERS = {'PR_ITEM': 0,
                 'PR_REVIEW': 0,
                 'PR_SET_HOLDING': 0,
                 'PR_UNSET_HOLDING': 0,
                 'PR_DELETE_ITEM': 0,
                 'PR_HIDE_ITEM': 0,
                 'PR_UNHIDE_ITEM': 0,
                 'PR_VENDOR': 0}

SIMULATION_PARAMETERS = {'N_EVENTS': 0,
                         'MAX_ITEMS': 0,
                         'MAX_DAYS': 0,
                         'HOLDING_PRICE_FACTOR': 0,
                         'RANDOM_SEED': False,
                         'USE_TRUE_POP': False,
                         'ABUNDANCE_INTERVAL_EST': 30}

SCRAPING_PARAMETERS = {'SCRAPE_ABSOLUTE_LIMIT': 0,
                       'SCRAPE_RELATIVE_LIMIT': 0,
                       'SCRAPE_TIME_INTERVAL': 0,
                       'SCRAPE_POP_DISCOVERY_BUDGET': 0,
                       'REQUEST_CAP': 0}

EXPECTED_VALUES = {'ITEMS': 123132, 'REVIEWS': 258184, 'DELETES': 36081, 'UNHIDES': 12577, 'HIDES': 15577, 'VENDORS': 3625}


class Marketplace:
    def __init__(self, pr_parameters, sim_parameters, scraper_parameters):
        self.random_seed =      sim_parameters['RANDOM_SEED']
        # id to identify specific instances, we will use this name to generate plots and create directories
        # ID should contain: date + seed, and its directory should print parameters used
        self.id =               str(datetime.now().strftime('%m.%d-%H:%M') + '_' +str(self.random_seed))
        # base path for the simulation, here we will store serialized data, plots, and configs
        self.base_path =        BASE_SIMULATION_RESULT_FOLDER + self.id +'/'
        self.serialized_path =  self.base_path + 'serialized/'
        # simulation parameters
        self.n_events =                         sim_parameters['N_EVENTS']
        self.max_items =                        sim_parameters['MAX_ITEMS']
        self.max_days =                         sim_parameters['MAX_DAYS']
        self.holding_price_factor =             sim_parameters['HOLDING_PRICE_FACTOR']
        self.abundance_estimation_interval =    sim_parameters['ABUNDANCE_INTERVAL_EST']
        # probability parameters
        self.pr_item =          pr_parameters['PR_ITEM']
        self.pr_review =        pr_parameters['PR_REVIEW']
        self.pr_set_holding =   pr_parameters['PR_SET_HOLDING']
        self.pr_unset_holding = pr_parameters['PR_UNSET_HOLDING']
        self.pr_delete_item =   pr_parameters['PR_DELETE_ITEM']
        self.pr_hide_item =     pr_parameters['PR_HIDE_ITEM']
        self.pr_unhide_item =   pr_parameters['PR_UNHIDE_ITEM']
        self.pr_vendor =        pr_parameters['PR_VENDOR']
        # optimization parameters
        self.acceptable_error_margin =      0.01
        self.grad_des_adj_factor_wide =     0.05
        self.grad_des_adj_factor_narrow =   0.01
        # empirical values
        self.exp_items =    EXPECTED_VALUES['ITEMS']
        self.exp_reviews =  EXPECTED_VALUES['REVIEWS']
        self.exp_deletes =  EXPECTED_VALUES['DELETES']
        self.exp_hides =    EXPECTED_VALUES['HIDES']
        self.exp_unhides =  EXPECTED_VALUES['UNHIDES']
        self.exp_vendors =  EXPECTED_VALUES['VENDORS']
        # empirical distributions
        self.exp_prices =               None
        self.exp_listing_popularity =   None
        self.exp_vendor_listings =      None
        # This is the same as the marketplace dictionary except we only append items and reviews to it
        # and never delete or hide anything or set holding prices for anything
        self.transcript = {'items': {}}
        # we just use a copy of a constant definition in case we need to alter it in the future
        # this is equivalent to command_frequency
        self.events = dict(MARKETPLACE_EVENTS)
        self.timed_events = {}
        # this used to be the 'marketplace' dictionary.
        # these are dictionaries containing items in a 'stateful' way
        self.items = {}
        self.items_timed = {}
        self.holding_items = {}
        self.hidden_items = {}
        self.deleted_items = {}
        #TEMPORARY - For debugging, can delete later
        self.population_per_interval = {}
        # vendors
        self.vendors = {}
        # scraper class
        self.scrapers = []
        # abundance data
        self.state_enc_history = {}
        self.abundances = {'STATE_JS': {}, 'STATE_SCHNABEL': {}, 'STATE_LP': {}, 'STATE': {}, 'TRANSCRIPT': {}}
        self.js_estimates = {}
        self.lp_estimates = {}
        self.schnabel_estimates = {}
        self.ab_estimation_days = []
        # supporting data
        self.vendor_item_assignment_prob_space = {}
        self.listing_fb_assignment_prob_space_by_i = {}
        self.listing_fb_assignment_prob_space_by_p = {}
        # we allow for multiple scrapers (with multiple configs to exist associated to a marketplace)
        for params in scraper_parameters:
            self.scrapers.append(Scraper(self.id, params, self.base_path))
        self.create_directories()
        self.serialize_marketplace_config(pr_parameters, sim_parameters, scraper_parameters)

    def set_empirical_distributions(self):
        utility.cprint('Loading empirical distributions...')
        with open(EMP_LISTING_PRICES, 'r') as fr:
            bk_listing_prices = pd.read_csv(fr)
        with open(EMP_LISTINGS_P_VENDOR,'r') as fr:
            bk_listings_per_vendor = pd.read_csv(fr)
        with open(EMP_FB_PER_LISTING,'r') as fr:
            bk_feedbacks_per_listing = pd.read_csv(fr)

        # Do price filtering to omit holding prices
        bk_listing_prices = bk_listing_prices[bk_listing_prices['price'] <= 250000]

        self.exp_prices = bk_listing_prices['price'].to_dict()
        self.exp_vendor_listings = bk_listings_per_vendor['id'].to_dict()
        self.exp_listing_popularity = bk_feedbacks_per_listing['count_feedbacks'].to_dict()

    def sampleNewEvent(self):

        num_items = len(self.items)
        num_holding_items = len(self.holding_items)
        num_hidden_items = len(self.hidden_items)

        num_unholding_items = num_items - num_holding_items
        num_unhidden_items = num_items - num_hidden_items

        # We are building up some numerical range and sub-dividing it into events. Then we select a random number and check which event it lands on to sample a marketplace event
        # This list stores the boundaries between these events so that we can easily check which one we got at the end
        probability_space = []
        total_probability = 0

        # Probability of a new item listing being created
        total_item_probability = self.pr_item
        total_probability += total_item_probability
        probability_space.append((total_probability, "NEW_ITEM"))

        # Probability of a review is the probability of reviewing any particular item times the number of items we have
        total_review_probability = self.pr_review * num_items
        total_probability += total_review_probability
        probability_space.append((total_probability, "REVIEW"))

        # Probabily of setting a holding a price is the probability of seting a holding price on any particular item times the number of items we have
        total_set_holding_probability = self.pr_set_holding * num_unholding_items
        total_probability += total_set_holding_probability
        probability_space.append((total_probability, "SET_HOLDING_PRICE"))

        # Probability of un-setting a holding price is the probability of unsetting a holding price for any particular item times the number of holding prices that we currently have
        total_unset_holding_probability = self.pr_unset_holding * num_holding_items
        total_probability += total_unset_holding_probability
        probability_space.append((total_probability, "UNSET_HOLDING_PRICE"))

        # Probability of deleting an item is the probability of deleting any particular item times the number of items that we have
        total_delete_probability = self.pr_delete_item * num_items
        total_probability += total_delete_probability
        probability_space.append((total_probability, "DELETE_ITEM"))

        # Probability of hiding an item is the probability of hiding any particular item times the number of items that we have
        total_hide_probability = self.pr_hide_item * num_unhidden_items
        total_probability += total_hide_probability
        probability_space.append((total_probability, "HIDE_ITEM"))

        # Probability of unhiding an item is the probability of unhiding any particular item times the number of hidden items that we have
        total_unhide_probability = self.pr_unhide_item * num_hidden_items
        total_probability += total_unhide_probability
        probability_space.append((total_probability, "UNHIDE_ITEM"))

        # Sample a random number and use it to determine what event took place
        sampled_num = np.random.sample() * total_probability

        # Determine what this random event corresponds to
        for event in probability_space:
            if sampled_num <= event[0]:
                # print("Sampled Number [" + str(sampled_num) + "] LEQ [" + str(event[0]) + "] Yielding Event Type [" + str(event[1]) + "]")
                return event[1]

        utility.cprint("Error: No event was selected?")
        return
    # When an item is bought, a quantity is specified. This function samples whatever distribution we use for this
    # Currently implementing this as a Binomial
    def sampleQuantity(self):
        # The binomial is tossing 20 coins, each with a 2% probabiliy of heads and sums them together
        return 1 + np.random.binomial(20, 0.02)

    def assign_feedback_to_listing(self):
        prob_list = list(self.listing_fb_assignment_prob_space_by_p.keys())
        total_probability = max(prob_list)
        ratio = np.random.sample()
        sampled_num = round(total_probability * ratio)
        try:
            closest_value = utility.take_closest(prob_list, sampled_num)
            item_id = self.listing_fb_assignment_prob_space_by_p[closest_value]
            a = self.items[item_id]
        except KeyError:
            print('Closest Value,', closest_value)
            print('Item ID,', item_id)

        return item_id

    # Generate a new review that could be for any non-hidden and non-holding price items
    # Draw the item to leave a review for according to a uniform sample over a probability space of weighted item popularities
    def generateNewReview(self, i, day_ct):
        # We can build a probability space over all of the different item listings to determine which listing the new review was generated for
        # We can also do this in another way, for instance, uniformly random
        listing_keys = list(self.items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)
        date_created = day_ct
        sampled_item = self.assign_feedback_to_listing()
        # Get the quantitiy of whatever item was just purchased
        quantity = self.sampleQuantity()
        current_price = self.items[sampled_item]['price']

        self.items[sampled_item]['reviews'].append((i, quantity, quantity * current_price, date_created))
        self.transcript['items'][sampled_item]['reviews'].append((i, quantity, quantity * current_price, date_created))
        return

    def sample_listings_per_vendor(self):
        return random.sample(list(self.exp_vendor_listings.values()), 1)[0]

    def generate_new_vendor(self, day_ct):
        # We can define a measure of popularity empirically or we can also use a well-known distribution
        popularity = np.random.zipf(3,1)[0]
        item_assignment_propensity = self.sample_listings_per_vendor()
        name_prefix = 'VENDOR-'
        vendor_keys_ct = len(self.vendors)
        name = name_prefix + str(vendor_keys_ct)
        # We create the probability space for this vendor to later sample from
        if len(self.vendor_item_assignment_prob_space) == 0:
            total_probability = item_assignment_propensity
        else:
            total_probability = list(self.vendor_item_assignment_prob_space.values())[-1] + item_assignment_propensity
        self.vendor_item_assignment_prob_space[name] = total_probability
        self.vendors[name] = {'popularity': popularity, 'first_seen': day_ct, 'items': {}, 'item_assignment_propensity': item_assignment_propensity}

    # We sample a vendor to assign to a listing that is created
    # To do this, we create a probability space based on a vendor's 'item_assignment_propensity'
    # The 'item_assignment_propensity' is calculated by using the empirical distribution of listing counts per vendor
    def assign_vendor_to_listing(self):
        #return random.sample(self.vendors.keys(), 1)[0]
        total_probability = max(list(self.vendor_item_assignment_prob_space.values()))
        sampled_num = total_probability * np.random.sample()
        for vendor, prob in self.vendor_item_assignment_prob_space.items():
            if sampled_num <= prob:
                return vendor

    def sample_price(self):
        return random.sample(list(self.exp_prices.values()), 1)[0]

    # We sample directly from the empirical distribution of listing popularities (determined by their feedback counts)
    def sample_listing_popularity(self):
        return random.sample(list(self.exp_listing_popularity.values()), 1)[0]

    # Generate a new item. This item can be parameterized in various ways including its popularity, price, and category
    def generateNewItem(self, item_id, day=-1):
        price = self.sample_price()
        category = 1
        vendor = self.assign_vendor_to_listing()
        popularity = self.sample_listing_popularity()
        if len(self.listing_fb_assignment_prob_space_by_p) == 0:
            total_popularity = 0
            total_probability = popularity
        else:
            total_popularity = list(self.listing_fb_assignment_prob_space_by_p.keys())[-1]
            total_probability = total_popularity + popularity

        self.listing_fb_assignment_prob_space_by_p[total_probability] = item_id
        self.items[item_id] = {'price': price, "popularity": popularity, "category": category, "reviews": [], "first_seen": day,
                                   "isHolding": False, 'vendor': vendor}
        self.transcript['items'][item_id] = {'price': price, "popularity": popularity, "category": category, "reviews": [], "first_seen": day,
                                  "isHolding": False, 'vendor': vendor}
        self.vendors[vendor]['items'][item_id] = {'price': price, "popularity": popularity, "category": category, "reviews": [], "first_seen": day,
                                   "isHolding": False, 'vendor': vendor}
        if day >= 0:
            self.items_timed[day][item_id] = self.items[item_id]

        return

    def flush_information_policy(self, day_ct, type='ITEMS', period_days=90):
        if (day_ct % period_days) == 0:
            if type == 'ITEMS':
                return
            if type == 'REVIEWS':
                return
            if type == 'VENDORS':
                return

    def setHoldingPrice(self, i):
        # Randomly select a listing and delete it from the items dictionary and move it into the hodling prices dictionary

        # Randomly select a listing
        listing_keys = list(self.items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)

        # Grab the listing from the normal list
        listing = self.items[listing_keys[sampled_num]]
        # Set the holding price of the listing to be 10 times the normal price
        listing['price'] = self.holding_price_factor * listing['price']
        listing['isHolding'] = True
        # Add the holding price item to the holding_price items dictionary
        self.holding_items[listing_keys[sampled_num]] = listing
        # Delete this item from the normal set of items
        del self.items[listing_keys[sampled_num]]

        return

    # Pick an item listing at uniform random from the list of items with holding prices and return it to its normal value
    # Remove it from the list of holding price items and return it to the general list of items
    def unsetHoldingPrice(self, i):
        # Randomly select a listing from the holding price listings
        listing_keys = list(self.holding_items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)

        # Grab the listing from the holding items
        listing = self.holding_items[listing_keys[sampled_num]]
        # Unset the holding price of this listing
        listing['price'] = listing['price'] / self.holding_price_factor
        listing['isHolding'] = False
        # Add the listing back to the normal dictionary
        self.items[listing_keys[sampled_num]] = listing
        # Delete this item from the holding price items
        del self.holding_items[listing_keys[sampled_num]]

        return

    # Pick an item listing at uniform random from the list of items and delete it
    def deleteItem(self, i):
        # Randomly select a listing
        listing_keys = list(self.items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)

        # Delete this item listing
        item_id = listing_keys[sampled_num]
        listing = self.items[item_id]
        vendor = self.items[item_id]['vendor']
        self.deleted_items[item_id] = listing
        del self.items[item_id]
        del self.vendors[vendor]['items'][item_id]
        #Note: Currently, when we delete an item from the probability space, we can unwillingly be increasing the prob space
        # of the subsequent item.
        self.listing_fb_assignment_prob_space_by_p = {key: val for key, val in self.listing_fb_assignment_prob_space_by_p.items() if val != item_id}
        return

    # Pick an item listing at uniform random from the list of items and move it to the list of hidden items
    def hideItem(self, i):
        # Randomly select a listing
        listing_keys = list(self.items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)

        # Grab the listing from the normal list
        listing = self.items[listing_keys[sampled_num]]
        # Add the hidden item item to the hidden items dictionary
        self.hidden_items[listing_keys[sampled_num]] = listing
        # Delete this item from the normal set of items
        item_id = listing_keys[sampled_num]
        vendor = self.items[item_id]['vendor']
        del self.items[listing_keys[sampled_num]]
        # Delete this item from the vendors' items
        del self.vendors[vendor]['items'][item_id]
        self.listing_fb_assignment_prob_space_by_p = {key: val for key, val in self.listing_fb_assignment_prob_space_by_p.items() if val != item_id}

        return

    # Pick an item listing at unform random from the list of hidden items and move it to the list of normal items
    def unhideItem(self, i):
        # Randomly select a listing from the holding price listings
        listing_keys = list(self.hidden_items.keys())
        num_listings = len(listing_keys)
        sampled_num = math.floor(np.random.sample() * num_listings)

        # Grab the listing from the holding items
        listing = self.hidden_items[listing_keys[sampled_num]]
        # Add the listing back to the normal dictionary
        item_id = listing_keys[sampled_num]
        self.items[listing_keys[sampled_num]] = listing
        # Add the listing back to the vendor's profile
        vendor = self.items[item_id]['vendor']
        self.vendors[vendor]['items'][item_id] = listing
        total_probability = self.items[item_id]['popularity'] + list(self.listing_fb_assignment_prob_space_by_p.keys())[-1]
        self.listing_fb_assignment_prob_space_by_i[total_probability] = item_id
        # Delete this item from the holding price items
        del self.hidden_items[listing_keys[sampled_num]]

        return

    def reset_marketplace(self):
        self.events = dict(MARKETPLACE_EVENTS)
        self.items = {}
        self.holding_items = {}
        self.hidden_items = {}
        self.items_timed = {}
        self.transcript = {"items": {}}

    def _get_sma_day_limit(self, day_ct):
        tolerance_bound_bot = 1.5
        tolerance_bound_top =  3
        # The SMA_DICTIONARY is a set of day/count pairs that defines the shape of the market. Our tolerance bounds
        # determine how much variance we want from this shape.
        with open(SMA_DICTIONARY, 'r') as fr:
            sma_fb = json.load(fr)

        average = list(sma_fb.values())[day_ct]
        day_event_limit = math.floor(random.uniform(tolerance_bound_bot * average, tolerance_bound_top * average))

        return day_event_limit

    def _adjust_param(self, given, expected, param):
        lower_bound = (1.0-self.acceptable_error_margin) * given
        upper_bound = (1.0+self.acceptable_error_margin) * given
        if lower_bound < expected < upper_bound:
            utility.cprint('Acceptable param!')
            return param, False
        else:
            if given > expected:
                adjustment = random.uniform(1.0-self.grad_des_adj_factor_wide, 1.0-self.grad_des_adj_factor_narrow)
                param *= adjustment
                utility.cprint("Adjusted by (-)", adjustment)
            elif given < expected:
                adjustment = random.uniform(1.0+self.grad_des_adj_factor_narrow, 1.0+self.grad_des_adj_factor_wide)
                param *= adjustment
                utility.cprint("Adjusted by (+)", adjustment)
        return param, True

    def _faux_grad_descent(self):
        reviews = self.events['REVIEW']
        deletes = self.events['DELETE_ITEM']
        hides = self.events['HIDE_ITEM']
        unhides = self.events['UNHIDE_ITEM']

        utility.cprint('Checking reviews...' + str(reviews))
        self.pr_review, adj_pr_review = self._adjust_param(reviews, self.exp_reviews, self.pr_review)
        utility.cprint('Checking deletes...' + str(deletes))
        self.pr_delete_item, adj_pr_delete = self._adjust_param(deletes, self.exp_deletes, self.pr_delete_item)
        utility.cprint('Checking hides...' + str(hides))
        self.pr_hide_item, adj_pr_hide = self._adjust_param(hides, self.exp_hides, self.pr_hide_item)
        utility.cprint('Checking unhides...' + str(unhides))
        self.pr_unhide_item, adj_pr_unhide = self._adjust_param(unhides, self.exp_unhides, self.pr_unhide_item)

        if adj_pr_review == adj_pr_delete == adj_pr_hide == adj_pr_unhide == False:
            utility.cprint('Converged!')
            utility.cprint("Results: REVIEW: ", self.pr_review, "DELETE: ", self.pr_delete_item, "HIDE: ", self.pr_hide_item,
                  "UNHIDE: ", self.pr_unhide_item)
            return True

        parameter_sum = self.pr_review + self.pr_delete_item + self.pr_hide_item + self.pr_unhide_item
        if parameter_sum > parameter_sum * 2:
            utility.cprint("Resetting parameters to avoid explosion...")
            self.pr_review =        3.972801523229621e-05
            self.pr_delete_item =   5.540931170095787e-06
            self.pr_hide_item =     2.506809116838376e-06
            self.pr_unhide_item =   5.815538075935886e-05

        return False

    def find_parameters(self):
        while True:
            converged = self.generate_marketplace(find_params=True)
            if converged == True:
                utility.cprint("PR REVIEW:", self.pr_review)
                utility.cprint("PR DELETE:", self.pr_delete_item)
                utility.cprint("PR HIDE:", self.pr_hide_item)
                utility.cprint("PR UNHIDE:", self.pr_unhide_item)
                break
            self.reset_marketplace()

    def record_event(self, event, i, day_ct):
        self.events[event] += 1
        if event == "NEW_ITEM":
            self.timed_events[day_ct]["NEW_ITEM"] += 1
            self.generateNewItem(i, day=day_ct)
        elif event == "REVIEW":
            self.timed_events[day_ct]["REVIEW"] += 1
            self.generateNewReview(i, day_ct)
        elif event == "SET_HOLDING_PRICE":
            self.timed_events[day_ct]["SET_HOLDING_PRICE"] += 1
            self.setHoldingPrice(i)
        elif event == "UNSET_HOLDING_PRICE":
            self.unsetHoldingPrice(i)
            self.timed_events[day_ct]["UNSET_HOLDING_PRICE"] += 1
        elif event == "DELETE_ITEM":
            self.deleteItem(i)
            self.timed_events[day_ct]["DELETE_ITEM"] += 1
        elif event == "HIDE_ITEM":
            self.hideItem(i)
            self.timed_events[day_ct]["HIDE_ITEM"] += 1
        elif event == "UNHIDE_ITEM":
            self.unhideItem(i)
            self.timed_events[day_ct]["UNHIDE_ITEM"] += 1

    def is_simulation_end(self, i, find_params):
        # We define a condition/s to stop our simulation. For example, we may choose to stop based on the number of items
        if self.events['NEW_ITEM'] > self.max_items * (1 + self.acceptable_error_margin):
            utility.cprint('Reached item limit...')
            self.print_progress(i)
            return True

        # We could also stop it based on the number of events
        number_of_events = self.events['NEW_ITEM'] + self.events['REVIEW'] + self.events['DELETE_ITEM'] + \
                            self.events['HIDE_ITEM'] + self.events['UNHIDE_ITEM'] + self.events['NEW_VENDOR']
        if number_of_events > self.n_events * (1 + self.acceptable_error_margin):
            utility.cprint('Reached event limit...')
            self.print_progress(i)
            return True

        # Or we could stop it based on the number of days elapsed
        if len(self.timed_events) > self.max_days * (1 + self.acceptable_error_margin):
            utility.cprint('Reached day limit...')
            self.print_progress(i)
            return True

        # For each case we can also define an acceptable_error_margin to introduce variation between the stops of various simulations

    def print_progress(self, i, t0=False):
        utility.cprint(str(i) + "- Command Freq: " + str(self.events), mute_all=False)
        if t0:
            t1 = time.clock() - t0
            utility.cprint('Round:', i, 'Seconds:', t1)

    def estimate_state_population(self, day_ct):
        ## POPULATION ESTIMATES FROM THE MARKETPLACE'S STATE
        enc_history_inp_file = self.base_path + 'serialized/' + self.id \
                               + '/encounter_histories/' + str(day_ct) + '.inp'
        r_simulation_env_path = self.base_path + 'serialized/' + self.id + '/R_env/'
        enc_hist_inp_file, enc_hist_inp_file_alt = utility.create_enc_history_file(self.state_enc_history, enc_history_inp_file)
        try:
            js_estimate = round(float(
                statistics.jolly_seber_estimate(enc_hist_inp_file, len(self.js_estimates), r_simulation_env_path)))
        except subprocess.CalledProcessError as e:
            utility.cprint('Convergence issue, retrying...', mute_all=False)
            js_estimate = round(float(
                statistics.jolly_seber_estimate(enc_hist_inp_file_alt, len(self.js_estimates), r_simulation_env_path)))
            utility.cprint(str(js_estimate) + r_simulation_env_path, mute_all=False)
        lp_estimate = round(statistics.lp_estimate(self.state_enc_history))
        schnabel_estimate = round(statistics.schnabel_estimate(enc_hist_inp_file))

        self.abundances['STATE_JS'][day_ct] = self.js_estimates[day_ct] = js_estimate
        self.abundances['STATE_LP'][day_ct] = self.lp_estimates[day_ct] = lp_estimate
        self.abundances['STATE_SCHNABEL'][day_ct] = self.schnabel_estimates[day_ct] = schnabel_estimate
        self.abundances['STATE'][day_ct] = len(self.items)
        self.abundances['TRANSCRIPT'][day_ct] = len(self.transcript['items'])

        return {'STATE_JS': js_estimate, 'STATE_LP': lp_estimate, 'STATE_SCHNABEL': schnabel_estimate, 'STATE': len(self.items), 'TOTAL': len(self.transcript['items'])}

    '''
    Print population calculation from a given scraper class as well as true market values
    '''
    def population_estimates(self, scraper, day_ct, plot=False):
        ## POPULATION ESTIMATES FROM OUR SCRAPER
        utility.cprint('Population Estimates')
        utility.cprint('JS Estimate: ' + str(list(scraper.js_estimates.values())[-1]))
        utility.cprint('Schnabel Estimate' + str(list(scraper.schnabel_estimates.values())[-1]))
        utility.cprint('LP Estimate: ' + str(list(scraper.lp_estimates.values())[-1]))
        utility.cprint('Current True Population: ' + str(len(self.items)))


        utility.cprint('Perfect JS Estimate: ' + str(list(self.js_estimates.values())[-1]))
        utility.cprint('Perfect LP Estimate: ' + str(list(self.lp_estimates.values())[-1]))
        utility.cprint('Perfect Schnabel Estimate: ' + str(list(self.schnabel_estimates.values())[-1]))

        scraper.abundances['STATE_JS'][day_ct] = list(self.js_estimates.values())[-1]
        scraper.abundances['STATE_LP'][day_ct] = list(self.lp_estimates.values())[-1]
        scraper.abundances['STATE_SCHNABEL'][day_ct] = list(self.schnabel_estimates.values())[-1]

        ## TRUE POPULATIONS
        utility.cprint('Current State True Population: ' + str(len(self.items)))
        utility.cprint('Transcript Total True Population: ' + str(len(self.transcript['items'])))
        scraper.abundances['STATE'][day_ct] = len(self.items)
        scraper.abundances['TOTAL'][day_ct] = len(self.transcript['items'])
        self.population_per_interval[day_ct] = len(self.items)


    def record_state_enc_history(self, day_ct):
        if day_ct % self.abundance_estimation_interval == 0:
            self.ab_estimation_days.append(day_ct)
            for item_id, item in self.items.items():
                if item_id not in self.state_enc_history:
                    self.state_enc_history[item_id] = {}
                self.state_enc_history[item_id][day_ct] = 1

            # for all the items we didn't see, we record a 0 in their encounter history
            for item_id in self.state_enc_history:
                for day_ct_ in self.ab_estimation_days:
                    if day_ct_ not in self.state_enc_history[item_id]:
                        self.state_enc_history[item_id][day_ct_] = 0

        return

    def serialize_marketplace_config(self, pr_parameters, sim_parameters, scraper_parameters):
        utility.cprint('Saving market config...')
        with open(self.base_path + 'simulation_config_summary.txt', 'w') as fw:
            fw.write('****SIMULATION SUMMARY****\n\n')
            fw.write('ID: ' + str(self.id)+'\n')
            fw.write('Seed: ' + str(self.random_seed)+'\n')
            for scraper in self.scrapers:
                fw.write('-------------------------------------\n')
                fw.write('Scraper ID: ' + str(scraper.id) + '\n')
                fw.write('Absolute Limit: ' + str(scraper.scrape_absolute_limit)+'\n')
                fw.write('Relative Limit:' + str(scraper.scrape_relative_limit)+'\n')
                fw.write('Scraping Interval: ' + str(scraper.scrape_timed_interval)+'\n')
                fw.write('Pop Discovery Budget: ' + str(scraper.scrape_pop_discovery_budget)+'\n')
                fw.write('Request Cap: ' + str(scraper.request_cap)+'\n')
        with open(self.base_path + 'pr_parameters.json', 'w') as fw:
            json.dump(pr_parameters, fw)
        with open(self.base_path + 'sim_parameters.json', 'w') as fw:
            json.dump(sim_parameters, fw)
        with open(self.base_path + 'scraper_parameters.json', 'w') as fw:
            json.dump(scraper_parameters, fw)

    def serialize_marketplace_state(self, with_scrapes):
        utility.cprint('Saving marketplace and scrapers\' state...')
        filepath = self.serialized_path + 'states/'
        with open(filepath + 'transcript.json', 'w') as fw:
            json.dump(self.transcript, fw)
        #with open(filepath + 'events.json', 'w') as fw:
        #    json.dump(self.events, fw)
        #with open(filepath + 'timed_events.json', 'w') as fw:
        #    json.dump(self.timed_events, fw)
        #with open(filepath + 'items.json', 'w') as fw:
        #    json.dump(self.items, fw)
        #with open(filepath + 'holding_items.json', 'w') as fw:
        #    json.dump(self.holding_items, fw)
        #with open(filepath + 'hidden_items.json', 'w') as fw:
        #    json.dump(self.hidden_items, fw)
        #with open(filepath + 'deleted_items.json', 'w') as fw:
        #    json.dump(self.hidden_items, fw)
        #with open(filepath + 'vendors.json', 'w') as fw:
        #    json.dump(self.vendors, fw)
        if with_scrapes:
            for scraper in self.scrapers:
                filepath = self.serialized_path + scraper.id + '/states/'
                #with open(filepath + 'scrape_dates.json', 'w') as fw:
                #    json.dump(scraper.scrape_dates, fw)
                #with open(filepath + 'scrapes_absolute.json', 'w') as fw:
                #    json.dump(scraper.scrapes_absolute, fw)
                #with open(filepath + 'scrapes_relative.json', 'w') as fw:
                #    json.dump(scraper.scrapes_relative, fw)
                #with open(filepath + 'scrapes_by_popularity.json', 'w') as fw:
                #    json.dump(scraper.scrapes_by_popularity, fw)
                #with open(filepath + 'failed_rescrapes.json', 'w') as fw:
                #    json.dump(scraper.failed_rescrapes, fw)
                #with open(filepath + 'rel_enc_history.json', 'w') as fw:
                #    json.dump(scraper.rel_enc_history, fw)
                #with open(filepath + 'js_estimates.json', 'w') as fw:
                #    json.dump(scraper.js_estimates, fw)
                #with open(filepath + 'lp_estimates.json', 'w') as fw:
                #    json.dump(scraper.lp_estimates, fw)
                #with open(filepath + 'schnabel_estimates.json', 'w') as fw:
                #    json.dump(scraper.schnabel_estimates, fw)
                #with open(filepath + 'page_requests.json', 'w') as fw:
                #    json.dump(scraper.page_requests, fw)
                with open(filepath + 'abundances.json', 'w') as fw:
                    json.dump(scraper.abundances, fw)

    def create_directories(self):
        utility.cprint('Initializing directories...')
        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        # We create the necessary folders
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        if not os.path.exists(self.serialized_path):
            os.makedirs(self.serialized_path)
        # Contains marketplace states
        if not os.path.exists(self.serialized_path + 'states/'):
            os.makedirs(self.serialized_path + 'states/')
        #if not os.path.exists(self.serialized_path + self.id + '/encounter_histories/'):
        #    os.makedirs(self.serialized_path + self.id + '/encounter_histories/')
        #if not os.path.exists(self.serialized_path + self.id + '/R_env/'):
        #    os.makedirs(self.serialized_path + self.id + '/R_env/')
        for scraper in self.scrapers:
            # Create a folder that will contained serialized data (e.g. txt, json, csv, etc.)
            if not os.path.exists(self.serialized_path + scraper.id):
                os.makedirs(self.serialized_path + scraper.id)
            # Encounter histories used for RMark, contains .inp files
            if not os.path.exists(self.serialized_path + scraper.id + '/encounter_histories'):
                os.makedirs(self.serialized_path + scraper.id + '/encounter_histories')
            # R environment where RMark files will be stored (e.g., model states, mark output files, logs, etc.)
            if not os.path.exists(self.serialized_path + scraper.id + '/R_env'):
                os.makedirs(self.serialized_path + scraper.id + '/R_env')
            # Will contain .json files with the scraper states
            if not os.path.exists(self.serialized_path + scraper.id + '/states'):
                os.makedirs(self.serialized_path + scraper.id + '/states')

    def generate_marketplace(self, with_scrape=True, find_params=False):
        #We set the empirical distributions for generating marketplace events
        self.set_empirical_distributions()
        do_revenue_calculations = True
        if find_params:
            with_scrape = False
        if self.random_seed is not False:
            np.random.seed(self.random_seed)
        if with_scrape is False:
            print('Scraping is disabled...')

        # This is the max number of events that will occur in a day
        day_event_limit = 0
        # This is a counter of the number of events that have occurred
        day_event_ct = 0
        # This is a counter that keeps track of where we are in terms of days
        day_ct = 0

        # Main loop where we simulate the marketplace process
        for i in range(self.n_events*2):
            # Start a day
            if day_event_ct == 0:
                # Set the limit of events for the day (we choose between an arbitrary range)
                # We set the number of events we want to simulate. Can be determined empirically or drawing from a distribution
                day_event_limit = self._get_sma_day_limit(day_ct)
                if day_event_limit < 1:
                    day_event_limit = 1
                self.timed_events[day_ct] = {       "NEW_ITEM": 0,
                                                    "REVIEW": 0,
                                                    "SET_HOLDING_PRICE": 0,
                                                    "UNSET_HOLDING_PRICE": 0,
                                                    "DELETE_ITEM": 0,
                                                    "HIDE_ITEM": 0,
                                                    "UNHIDE_ITEM": 0,
                                                    'NEW_VENDOR': 0}
                self.items_timed[day_ct] = {}

            # We specify the number of vendors we want to appear in the market in a given day
            if day_event_ct == 0:
                # We can again, use an empirical distribution or define this in another way. In the example below, we
                # are creating a constant number of vendors each day with some randomness
                for _ in range(0, random.randint(self.pr_vendor[0], self.pr_vendor[1]) + 1):
                    self.generate_new_vendor(day_ct)
                    self.events['NEW_VENDOR'] += 1
                    self.timed_events[day_ct]['NEW_VENDOR'] += 1

            # Sample a new item-related event
            event = self.sampleNewEvent()
            # Record the event that we drew
            self.record_event(event, i, day_ct)
            day_event_ct += 1


            # If we filled our day bucket with events, we move on to the next
            if day_event_ct == day_event_limit:
                utility.cprint('Day:' + str(day_ct))
                if with_scrape:
                    # Each scraper has a built-in conditional to check for its quota
                    for scraper in self.scrapers:
                        # We scrape the market in whatever way we see fit.
                        scraper.uniform_scrape(self.items, day_ct)
                        #scraper.scrape_with_abundance_estimation(self.items, day_ct)

                # We start another day, and we reset the event counter to 0
                day_ct += 1
                day_event_ct = 0
                # We reset the quotas of all the scrapers
                for scraper in self.scrapers:
                    scraper.reset_request_limits()

            # Show progress
            if (i % 10000) == 0:
                self.print_progress(i, t0=False)


            if self.is_simulation_end(i, find_params):
                # For coverage distribution experiments
                state_feedback_ct = 0
                transcript_feedback_ct = 0
                for item_id in self.items:
                    state_feedback_ct += len(self.items[item_id]['reviews'])
                for item_id in self.transcript['items']:
                    transcript_feedback_ct += len(self.transcript['items'][item_id]['reviews'])
                state_totals = {'vendors': len(self.vendors), 'items': len(self.items), 'feedbacks': state_feedback_ct}
                transcript_totals = {'vendors': len(self.vendors), 'items': len(self.transcript['items']),
                                     'feedbacks': transcript_feedback_ct}

                try:
                    print('Finished at time: {}'.format(datetime.now().strftime("%H:%M:%S")))
                except Exception as e:
                    pass
                for i, scraper in enumerate(self.scrapers):
                    self.serialize_marketplace_state(with_scrape)
                break


    def total_coverage_from_transcript(self):
        nr_reviews = 0
        # check for vendor coverage
        vendors = []
        for item_listing in self.transcript['items']:
            # print("Transcript Listing [" + str(item_listing) + "]  Num Reviews [" + str(len(transcript['items'][item_listing]['reviews'])) + "]")
            if self.transcript['items'][item_listing]['vendor'] not in vendors:
                vendors.append(self.transcript['items'][item_listing]['vendor'])
            nr_reviews += len(self.transcript['items'][item_listing]['reviews'])

        nr_vendors = len(vendors)
        coverage = nr_reviews + nr_vendors
        return coverage

    def total_coverage_from_state(self):
        nr_reviews = 0
        vendors = []
        for item_id, item in self.items.items():
            if item['vendor'] not in vendors:
                vendors.append(item['vendor'])
            nr_reviews += len(item['reviews'])

        for item_id, item in self.holding_items.items():
            if item['vendor'] not in vendors:
                vendors.append(item['vendor'])
            nr_reviews += len(item['reviews'])

        for item_id, item in self.hidden_items.items():
            if item['vendor'] not in vendors:
                vendors.append(item['vendor'])
            nr_reviews += len(item['reviews'])

        nr_vendors = len(vendors)
        coverage = nr_vendors + nr_reviews
        return coverage


class Scraper:
    def __init__(self, id, scraper_parameters, simulation_base_path):
        # parameters
        # Request limit for uniform scraper
        self.scrape_absolute_limit = scraper_parameters['SCRAPE_ABSOLUTE_LIMIT']
        # Request limit for uniform scraper + abundance estimation
        self.scrape_relative_limit = scraper_parameters['SCRAPE_RELATIVE_LIMIT']
        # % of the requests dedicated to discovery for the popularity scraper
        self.scrape_pop_discovery_budget = scraper_parameters['SCRAPE_POP_DISCOVERY_BUDGET']
        # Number of days between each scrape
        self.scrape_timed_interval = scraper_parameters['SCRAPE_TIME_INTERVAL']
        self.count_holding = False
        self.quantity_coefficient = 1
        self.request_cap = scraper_parameters['REQUEST_CAP']
        self.scrape_dates = []
        # scraper data
        self.scrapes_absolute = {}
        self.scrapes_relative = {}
        self.scrapes_by_popularity = {}
        # used by popularity scraper
        self.failed_rescrapes = []
        # encounter history used for abundance estimation
        self.rel_enc_history = {}
        self.js_estimates, self.lp_estimates, self.schnabel_estimates = {}, {}, {}
        # track page requests per day
        self.page_requests = {'ABS': {}, 'REL': {}, 'POP': {}}
        # track and enforce request limits
        self.request_limit = {'ABS': 0, 'REL': 0, 'POP': 0}
        # track scraper coverage by scraper_key:
        self.abundances = {'JS': {}, 'Schnabel': {}, 'LP': {},'STATE': {}, 'TOTAL':{}, 'STATE_JS':{}, 'STATE_SCHNABEL':{}, 'STATE_LP':{}}
        # Same ID as the marketplace it belongs to + the request limit
        # There's a bug whereby if the path strings are too long the Rscript fails because Fortran apparently can't handle the strings' length
        #self.id = id + '_' + str(self.scrape_timed_interval) + 'days_' + str(self.request_cap) + 'reqs' + '_A0{}'.format(str(self.scrape_absolute_limit).split('.')[1]) #+ '_P0{}'.format(str(self.scrape_pop_discovery_budget))
        self.id = str(self.scrape_timed_interval) + 'd_' + str(self.request_cap) + 'r' + '_A0{}'.format(str(self.scrape_absolute_limit).split('.')[1]) + '_P0{}'.format(str(self.scrape_pop_discovery_budget).split('.')[1])
        self.simulation_base_path = simulation_base_path
        # ARCHIVED
        self.scrapes_timed = {}
        self.scrapes_timed_dedup = {}

    def reset_request_limits(self):
        self.request_limit = {'ABS': 0, 'REL': 0, 'POP': 0}

    def scraper_not_hit_quota(self, name):
        return True if self.request_limit[name] >= self.request_cap else False

    def scrape_error(self):
        error_probability = 0.05
        return np.random.binomial(1, error_probability)

    def uniform_scrape(self, items, day_ct):
        '''
        :param items: the item list from the Market simulation
        :param day_ct: the day we are in the simulation
        :return: none

        This artificial scraper will uniformly scrape.
        '''
        if (day_ct % self.scrape_timed_interval) != 0:
            return
        if day_ct not in self.page_requests['ABS']:
            self.page_requests['ABS'][day_ct] = 0
        absolute_limit = round(self.scrape_absolute_limit * len(items))
        absolute_limit = self.request_cap if absolute_limit > self.request_cap else absolute_limit
        # While we haven't hit our daily limit, we will keep scraping
        while self.request_limit['ABS'] < absolute_limit:
            # Uniform sampling of viewable items
            scraped_items = random.sample(items.keys(), len(items))
            for item_id in scraped_items:
                if self.request_limit['ABS'] >= absolute_limit:
                    break
                # We iterate through the listings but we have a probability that the scraper will have an error and not get a page
                if self.scrape_error():
                    self.request_limit['ABS'] += 1
                    # track page requests
                    self.page_requests['ABS'][day_ct] += 1
                    continue
                else:
                    self.request_limit['ABS'] += 1
                    # track page requests
                    self.page_requests['ABS'][day_ct] += 1
                    self.scrapes_absolute[item_id] = copy.deepcopy(items[item_id])

            if self.request_limit['ABS'] >= absolute_limit:
                break


    def scrape_with_abundance_estimation(self, items, day_ct):
        '''
        :param items: the item list from the Market simulation
        :param day_ct: the day we are in the simulation
        :return: none

        This artificial scraper will uniformly scrape and then after enough captures will produce abundance estimates.
        '''
        if (day_ct % self.scrape_timed_interval) != 0:
            return
        if day_ct not in self.page_requests['REL']:
            self.page_requests['REL'][day_ct] = 0
        self.scrape_dates.append(day_ct)
        relative_limit = self.request_cap
        # Can only start running abundance estimate with >2 captures
        lp_estimate, js_estimate, schnabel_estimate = 0,0,0
        # Abundance estimators based on the relative scrape
        if day_ct > self.scrape_timed_interval * 2:
            enc_history_inp_file = self.simulation_base_path + 'serialized/' + self.id \
                                   + '/encounter_histories/' + str(day_ct) + '.inp'
            r_simulation_env_path = self.simulation_base_path + 'serialized/' + self.id + '/R_env/'
            enc_hist_inp_file, enc_hist_inp_file_alt = utility.create_enc_history_file(self.rel_enc_history, enc_history_inp_file)
            # We need to pass the len of js_estimates so that we can find the previous file and load the previous model
            # RMark names the model files 'mark001.*'
            try:
                js_estimate = round(float(statistics.jolly_seber_estimate(enc_hist_inp_file, len(self.js_estimates), r_simulation_env_path)))
            except subprocess.CalledProcessError as e:
                print(e)
                utility.cprint('Convergence issue, retrying...', mute_all=False)
                try:
                    js_estimate = round(float(statistics.jolly_seber_estimate(enc_hist_inp_file_alt, len(self.js_estimates), r_simulation_env_path)))
                    utility.cprint(str(js_estimate) + r_simulation_env_path, mute_all=False)
                except subprocess.CalledProcessError as e:
                    print(e)
                    js_estimate = -1

            lp_estimate = round(statistics.lp_estimate(self.rel_enc_history))
            schnabel_estimate = round(statistics.schnabel_estimate(enc_hist_inp_file))
            self.js_estimates[day_ct] = js_estimate
            self.schnabel_estimates[day_ct] = schnabel_estimate
            self.lp_estimates[day_ct] = lp_estimate
            # Once we have an abundance estimation, we use that instead of just hitting our request_cap. We use JS by default
            relative_limit = int(js_estimate)
        self.abundances['JS'][day_ct] = js_estimate
        self.abundances['Schnabel'][day_ct] = schnabel_estimate
        self.abundances['LP'][day_ct] = lp_estimate

        # While we haven't hit our daily limit, we will keep scraping
        while self.request_limit['REL'] < relative_limit:
            # Uniform sampling of viewable items
            scraped_items = random.sample(items.keys(), len(items))
            for item_id in scraped_items:
                if self.request_limit['REL'] >= relative_limit:
                    break
                # We iterate through the listings but we have a probability that the scraper will have an error and not get a page
                if self.scrape_error():
                    self.request_limit['REL'] += 1
                    self.page_requests['REL'][day_ct] +=1
                    continue
                else:
                    # If there's no error, we got the item page
                    self.scrapes_relative[item_id] = copy.deepcopy(items[item_id])
                    # We record encounters for abundance estimation
                    # Set up encounter history for relative scrapes
                    if item_id not in self.rel_enc_history:
                        self.rel_enc_history[item_id] = {}
                    self.rel_enc_history[item_id][day_ct] = 1
                    self.request_limit['REL'] += 1
                    self.page_requests['REL'][day_ct] +=1
            if self.request_limit['REL'] >= relative_limit:
                break
        # for all the items we didn't see, we record a 0 in their encounter history
        for item_id in self.scrapes_relative:
            for day_ct_ in self.scrape_dates:
                if day_ct_ not in self.rel_enc_history[item_id]:
                    self.rel_enc_history[item_id][day_ct_] = 0
            # because we are adding 0s like this, the keys may get out of order, we will sort this when organizing the M array

        total_items = 0
        for item_id in self.rel_enc_history:
            total_items += self.rel_enc_history[item_id][day_ct]

    def popularity_scrape(self, items, day_ct):
        if (day_ct % self.scrape_timed_interval) != 0:
            return
        if day_ct not in self.page_requests['POP']:
            self.page_requests['POP'][day_ct] = 0
        # if there are less items than our daily request cap, we simply do discovery
        if len(items) <= self.request_cap:
            discovery_budget = self.request_cap
            rescrape_budget = -1
        else:
            discovery_budget = round(self.scrape_pop_discovery_budget * self.request_cap)
            rescrape_budget = round((1 - self.scrape_pop_discovery_budget) * self.request_cap)

        # rescrape the top X items (they will already be ordered from the previous iteration)
        rescraped_pages = 0
        # we keep track of leftover_budget and apply it to discovery later
        leftover_budget = 0
        while rescraped_pages <= rescrape_budget:
            try:
                item_id_to_rescrape = list(self.scrapes_by_popularity.keys())[rescraped_pages]
            except IndexError:
                break
            # if the item is in our failed list we don't rescrape it
            if item_id_to_rescrape in self.failed_rescrapes:
                leftover_budget += 1
            elif item_id_to_rescrape in items:
                self.scrapes_by_popularity[item_id_to_rescrape] = copy.deepcopy(items[item_id_to_rescrape])
            # if the item is not in the list of items then it must be hidden/deleted, we record that so we dont rescrape in the future
            else:
                self.failed_rescrapes.append(item_id_to_rescrape)
            rescraped_pages += 1
            if rescraped_pages >= rescrape_budget:
                break

        leftover_budget = leftover_budget + (rescrape_budget - rescraped_pages)
        discovered_pages = 0
        while discovered_pages <= discovery_budget + leftover_budget:
            expected_discovered_items = round(np.random.sample() * len(items))
            discovered_items = random.sample(items.keys(), expected_discovered_items)
            # we save the newly discovered items
            for disc_item_id in discovered_items:
                self.scrapes_by_popularity[disc_item_id] = copy.deepcopy(items[disc_item_id])
                # if we find again an item that we thought was deleted, we take it out of the failed list
                if disc_item_id in self.failed_rescrapes:
                    self.failed_rescrapes.remove(disc_item_id)
                discovered_pages += 1
                if discovered_pages >= discovery_budget:
                    break
            if discovered_pages >= discovery_budget:
                break

        self.request_limit['POP'] = rescraped_pages + discovered_pages
        self.page_requests['POP'][day_ct] = rescraped_pages + discovered_pages
        # sort items by review count
        self.scrapes_by_popularity = \
            {k: v for k, v in
             sorted(self.scrapes_by_popularity.items(), key=lambda item: len(item[1]['reviews']), reverse=True)}

    def total_sales_from_scrapes(self, scrape, count_holding=False, quantity_coefficient=1):
        revenue = 0
        for item_listing in scrape:
            listing = scrape[item_listing]
            listing_price = listing['price']
            isHolding = listing['isHolding']

            if isHolding == False or count_holding == True:
                for review in listing['reviews']:
                    revenue += listing_price * quantity_coefficient
            else:
                # print("Hodling price detected, we skip those")
                revenue += 0
        return revenue

    def total_coverage_from_scrapes(self, scrape, count_holding=False, quantity_coefficient=1):
        nr_reviews = 0
        vendors = []
        for item_listing in scrape:
            listing = scrape[item_listing]
            listing_price = listing['price']
            isHolding = listing['isHolding']
            vendor = listing['vendor']
            if vendor not in vendors:
                vendors.append(vendor)

            for _ in listing['reviews']:
                nr_reviews += 1
        nr_vendors = len(vendors)
        coverage = nr_vendors + nr_reviews

        return coverage


if __name__ == '__main__':
    processes = []
    # Parameters which determine behaviors relating to probabilities
    pr_parameters = dict(PR_PARAMETERS)
    pr_parameters['PR_ITEM'] =          1
    pr_parameters['PR_REVIEW'] =        4.301980431480647e-05
    pr_parameters['PR_SET_HOLDING'] = 0
    pr_parameters['PR_UNSET_HOLDING'] = 0
    pr_parameters['PR_DELETE_ITEM'] =   5.9986568481287185e-06
    pr_parameters['PR_HIDE_ITEM'] =     2.7187005317347344e-06
    pr_parameters['PR_UNHIDE_ITEM'] =   6.807406006357771e-05
    pr_parameters['PR_VENDOR'] = (1, 8)  # range for uniform distribution of vendor appearance

    # Other marketplace parameters
    sim_parameters = dict(SIMULATION_PARAMETERS)
    sim_parameters['N_EVENTS'] = 445511
    sim_parameters['MAX_ITEMS'] = 123132
    sim_parameters['MAX_DAYS'] = 700
    sim_parameters['HOLDING_PRICE_FACTOR'] = 1
    sim_parameters['RANDOM_SEED'] = 0
    sim_parameters['USE_TRUE_POP'] = True

    scraper_parameters_list = []
    scraper_parameters = dict(SCRAPING_PARAMETERS)
    # ABS Limits to test: 0.3, 0.6, 0.9
    scraper_parameters['SCRAPE_ABSOLUTE_LIMIT'] = 0.9
    scraper_parameters['SCRAPE_RELATIVE_LIMIT'] = 'JS Estimator'
    # Intervals to test: 30, 90, 180, maybe add logic for one-time scrapes towards the end of the market
    scraper_parameters['SCRAPE_TIME_INTERVAL'] = 15
    # It's best if we match the intervals for abundance estimation with the scrape interval for plots
    sim_parameters['ABUNDANCE_INTERVAL_EST'] = 15
    scraper_parameters['SCRAPE_POP_DISCOVERY_BUDGET'] = 0.75
    # 1 per second = 86,400 (60*60*24)
    # 1 per 5 seconds = 28,800
    # 1 per 10 seconds = 8,640
    # 1 per 20 seconds = 4,320
    scraper_parameters['REQUEST_CAP'] = 28800
    # We can add other kinds of scraper by following this format
    scraper_parameters_list.append(dict(scraper_parameters))

    # Or we can create other markets
    market = Marketplace(pr_parameters, sim_parameters, scraper_parameters_list)
    market.generate_marketplace()
