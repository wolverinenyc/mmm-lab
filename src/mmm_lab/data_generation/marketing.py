"""
Add marketing spend and effects to baseline geo data.
"""


# ## Key Design Points

# 1. **Geo-Level Transformations**: Each geo gets its own adstock and saturation applied to its allocated spend
#    - Ready for geo experiments (treatment vs control)
#    - Captures heterogeneity if geos have different spend levels

# 2. **K in Dollar Terms**: 
#    - TV: K=$5,000 means at $5k/week/geo, you're 50% saturated
#    - Paid Search: K=$2,000 means it saturates faster

# 3. Effect Calculation:
#    Effect = beta × pop_weight × geo_effectiveness × hill(adstock)
#    (standard formulation: beta is the national bookings ceiling)

import numpy as np
import pandas as pd


def geometric_adstock(spend, decay_rate):
    """
    Apply geometric adstock transformation.
    
    Args:
        spend: Array of spend values over time (n_weeks,)
        decay_rate: Carryover rate (0-1). E.g., 0.5 means 50% carries over
        
    Returns:
        adstocked: Array of adstocked spend values (n_weeks,)
        
    Example:
        spend = [100, 0, 0, 0]
        adstocked with decay=0.6: [100, 60, 36, 21.6]
    """
    n_weeks = len(spend)
    adstocked = np.zeros(n_weeks)
    adstocked[0] = spend[0]
    
    for t in range(1, n_weeks):
        adstocked[t] = spend[t] + decay_rate * adstocked[t-1]
    
    return adstocked


def hill_saturation(x, K, S):
    """
    Apply Hill saturation curve.
    
    Args:
        x: Input values (e.g., adstocked spend) - can be array or scalar
        K: Half-saturation point (x value where saturation = 0.5)
        S: Shape parameter (>0, typically 0.5-2.0)
           S=1: standard diminishing returns
           S>1: S-curve (initially accelerating then diminishing)
           S<1: very rapid diminishing returns
        
    Returns:
        Saturated values between 0 and 1
        
    Example:
        At x=K: hill_saturation(K, K, S) = 0.5 for any K, S
        At x=2*K with S=1: ≈ 0.67
        At x=5*K with S=1: ≈ 0.83
    """
    return (x ** S) / (x ** S + K ** S)


def calculate_adstock_multiplier(decay_rate, max_lags=52):
    """
    Calculate the effective spend multiplier from adstock.
    
    For spend of $1, the total effect over time is:
    1 + decay + decay^2 + decay^3 + ... ≈ 1/(1-decay)
    
    Args:
        decay_rate: Adstock decay rate
        max_lags: Maximum lags to consider (default 52 weeks = 1 year)
    
    Returns:
        Multiplier showing total cumulative effect
    """
    if decay_rate >= 1:
        return np.inf
    # Geometric series sum: 1 + r + r^2 + ... ≈ 1/(1-r)
    # But we'll calculate exactly up to max_lags
    return sum(decay_rate ** i for i in range(max_lags))


def add_marketing_effects(
    baseline_df,
    channels=['tv', 'paid_search']
):
    """
    Add marketing spend and incremental bookings to baseline geo data.
    
    MVP version:
    - Exogenous spend (predetermined national budgets)
    - Uncorrelated channels
    - National spend distributed to geos by population
    - Geo-level adstock and saturation transformations
    - Ground truth parameters for validation
    
    Args:
        baseline_df: DataFrame from generate_baseline_geo_data()
        channels: List of channel names
        
    Returns:
        DataFrame with additional columns:
            - spend_{channel}: Weekly spend per channel per geo
            - effect_{channel}: Incremental bookings per channel per geo  
            - total_bookings: baseline_bookings + sum(effects)
    """
    df = baseline_df.copy()
    
    n_weeks = df['date'].nunique()
    n_geos = df['geo'].nunique()
    
    # Get geo populations for spend allocation
    geo_populations = df.groupby('geo')['population'].first().sort_index().values
    pop_weights = geo_populations / geo_populations.sum()
    
    # Ground truth parameters for each channel
    # These match research literature (Google 2017, 2024 papers)
    channel_params = {
        'tv': {
            'weekly_budget': 100_000,
            'budget_variation': 0.2,
            'geo_budget_variation': 0.15,
            'adstock_rate': 0.5,
            'saturation_K': 5_000,
            'saturation_S': 1.0,
            'beta': 201_000,   # national bookings ceiling: beta * avg_hill / national_spend ≈ ROAS 1.08
            'geo_effectivness_cv': 0.2
        },
        'paid_search': {
            'weekly_budget': 80_000,
            'budget_variation': 0.2,
            'geo_budget_variation': 0.15,
            'adstock_rate': 0.3,
            'saturation_K': 2_000,
            'saturation_S': 1.5,
            'beta': 293_000,   # national bookings ceiling: beta * avg_hill / national_spend ≈ ROAS 2.28
            'geo_effectivness_cv': 0.25
        }
    }

    
    # Generate spend and effects for each channel
    geo_effectiveness = {}
    for channel in channels:
        params = channel_params[channel]
        cv = params['geo_effectivness_cv']

        # log-normal distribution ensures positive values 
        # CV controls spread around mean of 1 

        sigma = np.sqrt(np.log(cv**2 + 1))
        mu = -0.5 * sigma**2  # adjust so mean = 1.0 
        geo_mult = np.random.lognormal(mean=mu, sigma=sigma, size=n_geos)

        # normalize to mean 1.0
        geo_mult = geo_mult / geo_mult.mean() 
        geo_effectiveness[channel] = geo_mult


    for channel in channels: 
        params = channel_params[channel] 
        # Step 1: Generate national weekly spend with random variation
        national_spend = params['weekly_budget'] * np.random.uniform(
            1 - params['budget_variation'],
            1 + params['budget_variation'],
            n_weeks
        )
        
        geo_budget_mult = np.random.uniform(
            1 - params['geo_budget_variation'],
            1 + params['geo_budget_variation'],
            n_geos
        )

        geo_budget_mult = geo_budget_mult / geo_budget_mult.mean()


        # Step 2: Apply transformations at geo level
        spend_col = []
        effect_col = []
        
        for geo_idx in range(n_geos):
            geo_spend = national_spend * pop_weights[geo_idx] * geo_budget_mult[geo_idx]
            
            # Apply adstock (carryover effects)
            geo_adstocked = geometric_adstock(geo_spend, params['adstock_rate'])
            
            # Apply saturation (diminishing returns)
            geo_saturated = hill_saturation(
                geo_adstocked,
                K=params['saturation_K'],
                S=params['saturation_S']
            )
            
            # Calculate incremental effect: beta × pop_weight × geo_effectiveness × hill(adstock)
            geo_effectiveness_mult = geo_effectiveness[channel][geo_idx]

            geo_effect = (
                params['beta'] *            # bookings ceiling (national, scaled by population)
                pop_weights[geo_idx] *      # distribute by geo population
                geo_effectiveness_mult *    # geo-specific multiplier
                geo_saturated               # hill(adstock) ∈ [0, 1]
            )
            
            # Store results
            spend_col.extend(geo_spend)
            effect_col.extend(geo_effect)
        
        # Add to dataframe
        df[f'spend_{channel}'] = spend_col
        df[f'effect_{channel}'] = effect_col
    
    # Calculate total bookings
    effect_cols = [f'effect_{channel}' for channel in channels]
    df['total_bookings'] = df['baseline_bookings'] + df[effect_cols].sum(axis=1)
    
    # Calculate and display ground truth metrics
    print("\n" + "="*80)
    print("GROUND TRUTH MARKETING EFFECTS")
    print("="*80)
    
    for channel in channels:
        params = channel_params[channel]
        total_spend = df[f'spend_{channel}'].sum()
        total_effect = df[f'effect_{channel}'].sum()
        overall_roas = total_effect / total_spend if total_spend > 0 else 0

        # Store the calculated ROAS in params
        channel_params[channel]['overall_roas'] = round(overall_roas,3)

        # Also calculate average saturation to understand how saturated we are
        avg_saturation = df.groupby('geo').apply(
            lambda x: hill_saturation(
                geometric_adstock(x[f'spend_{channel}'].values, params['adstock_rate']),
                K=params['saturation_K'],
                S=params['saturation_S']
            ).mean()
        ).mean()
        
        print(f"\n{channel.upper()}:")
        print(f"  Total Spend:        ${total_spend:,.0f}")
        print(f"  Total Effect:       {total_effect:,.0f} bookings")
        print(f"  Overall ROAS:       ${overall_roas:.2f}")
        print(f"  Avg Saturation:     {avg_saturation:.1%}")
        print(f"  Adstock Rate:       {params['adstock_rate']:.1%} (carryover)")
        print(f"  Half-Sat Point (K): ${params['saturation_K']:,.0f}")
        print(f"  Shape (S):          {params['saturation_S']:.1f}")
    
    print("\n" + "="*80)
        
    # Store parameters including geo-specific multipliers
    df.attrs['channel_params'] = channel_params
    df.attrs['geo_effectiveness'] = geo_effectiveness  # ADD THIS LINE

    return df


if __name__ == "__main__":
    # Test adstock
    print("Testing Adstock:")
    spend = np.array([100, 0, 0, 0, 0])
    adstocked = geometric_adstock(spend, decay_rate=0.5)
    print(f"Original:  {spend}")
    print(f"Adstocked: {np.round(adstocked, 1)}")
    
    # Test Hill
    print("\nTesting Hill Saturation:")
    spend_levels = np.array([0, 1000, 2500, 5000, 10000])
    saturated = hill_saturation(spend_levels, K=2500, S=1.0)
    print(f"Spend:      {spend_levels}")
    print(f"Saturated:  {np.round(saturated, 3)}")
    print(f"At K=2500:  {hill_saturation(2500, K=2500, S=1.0):.3f} (should be 0.5)")