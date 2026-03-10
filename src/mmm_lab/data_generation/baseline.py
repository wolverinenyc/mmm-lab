"""
Generate baseline geo-level demand data (no marketing, just fundamentals).
"""

import numpy as np
import pandas as pd


def generate_baseline_geo_data(n_geos, n_weeks, start_date='2023-01-01', config=None):
    """
    Generate baseline geo-level bookings with seasonality, trends, and AR(1) noise.
    
    Geos are grouped by:
    - Region (geography): ~5 geos per region, share regional seasonality
    - Size tier (population): ~5 geos per tier, share size-based seasonality
    
    This creates natural comparison groups for synthetic control / geo experiments.
    
    Args:
        n_geos: Number of geos
        n_weeks: Number of weeks
        start_date: Start date string
        config: Optional dict to override defaults
    
    Returns:
        DataFrame with columns: date, geo, region, size_tier, population, baseline_bookings, market_conditions
    """
    # Default config
    cfg = {
        'population_range': (500_000, 3_000_000),
        'baseline_scaling_range': (0.0008, 0.0012),
        'growth_rate_range': (-0.05, 0.15),
        'seasonal_amplitude_range': (0.25, 0.35),
        'national_seasonal_amplitude': 0.30,
        'regional_seasonal_amplitude': 0.20,
        'size_seasonal_amplitude': 0.15,
        'ar_phi': 0.7,
        'ar_sigma': 0.08
    }
    if config:
        cfg.update(config)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_weeks, freq='W-MON')
    t = np.arange(n_weeks)
    
    # National seasonality (shared across all geos)
    national_seasonal = 1 + cfg['national_seasonal_amplitude'] * np.sin(2 * np.pi * t / 52 - np.pi/2)
    
    # Fixed geo characteristics
    geo_population = np.random.uniform(cfg['population_range'][0], cfg['population_range'][1], n_geos)
    geo_baseline_scaling = np.random.uniform(cfg['baseline_scaling_range'][0], cfg['baseline_scaling_range'][1], n_geos)
    geo_baseline_level = geo_population * geo_baseline_scaling  # baseline tied to population
    geo_growth_rate = np.random.uniform(cfg['growth_rate_range'][0], cfg['growth_rate_range'][1], n_geos)
    geo_seasonal_amplitude = np.random.uniform(cfg['seasonal_amplitude_range'][0], cfg['seasonal_amplitude_range'][1], n_geos)
    
    # Assign geos to regions (geography-based grouping)
    # For n_geos=20: n_regions=4, so ~5 geos per region
    n_regions = max(1, n_geos // 5)
    geo_region = np.repeat(np.arange(n_regions), np.ceil(n_geos / n_regions))[:n_geos]
    np.random.shuffle(geo_region)  # randomize assignment
    
    # Generate regional seasonality patterns
    # REASONING: Within US, regions share same holidays/calendar but differ in:
    #   1. TIMING: Minor differences (±2 weeks for spring break, back-to-school)
    #   2. AMPLITUDE: Some regions more seasonal than others (Minnesota vs Florida)
    regional_seasonal = {}
    for region_idx in range(n_regions):
        # Phase shift: ±2 weeks (NOT ±3 months - we're modeling US DMAs, not international)
        # ±2 weeks captures: Florida spring break earlier, varying back-to-school timing
        phase_shift = np.random.uniform(-np.pi/26, np.pi/26)  # 2π/52 weeks × 2 = π/26
        
        # Amplitude multiplier: Regional variation in HOW MUCH seasonality matters
        # Example: Florida (tourism, retirees) less seasonal than Minnesota (harsh winters)
        # Range: 0.8x to 1.2x means ±20% variation in seasonal amplitude
        amplitude_mult = np.random.uniform(0.8, 1.2)
        
        # IMPLICATION: Geos in same region will be highly correlated (critical for synthetic control)
        # but still have realistic differences in seasonal strength
        regional_seasonal[region_idx] = 1 + cfg['regional_seasonal_amplitude'] * amplitude_mult * np.sin(2 * np.pi * t / 52 - np.pi/2 + phase_shift)
    
    # Assign geos to size tiers (population-based grouping)
    # Larger populations → lower tier index (tier 0 = largest)
    geo_population_ranks = np.argsort(geo_population)
    n_size_tiers = max(1, n_geos // 5)
    geo_size_tier = np.zeros(n_geos, dtype=int)
    for idx, geo_idx in enumerate(geo_population_ranks):
        geo_size_tier[geo_idx] = min(idx // max(1, n_geos // n_size_tiers), n_size_tiers - 1)
    
    # Generate size-tier seasonality patterns
    # REASONING: Larger markets have MORE DIVERSIFIED economies
    #   - NYC: finance + tourism + tech + retail → less seasonal (%)
    #   - Small town: agriculture + one factory → more seasonal (%)
    # BUT: Larger markets have BIGGER ABSOLUTE SWINGS
    #   - NYC: ±20% of 2000 = ±400 bookings
    #   - Small: ±40% of 500 = ±200 bookings
    size_tier_seasonal = {}
    for tier_idx in range(n_size_tiers):
        # Phase shift: Minimal (±1 week) - size doesn't meaningfully affect WHEN peaks occur
        phase_shift = np.random.uniform(-np.pi/52, np.pi/52)
        
        # Amplitude multiplier: LARGER markets = SMALLER percentage seasonality
        # tier_idx increases with SMALLER geos (tier 0 = largest, tier 3 = smallest)
        # We want: tier 0 (large) → low amplitude, tier 3 (small) → high amplitude
        amplitude_mult = 0.7 + (tier_idx / max(1, n_size_tiers - 1)) * 0.6
        # tier 0 (largest):  0.7 + 0.0 = 0.7 → 70% amplitude → less % seasonal
        # tier 1:            0.7 + 0.2 = 0.9 → 90% amplitude
        # tier 2:            0.7 + 0.4 = 1.1 → 110% amplitude
        # tier 3 (smallest): 0.7 + 0.6 = 1.3 → 130% amplitude → more % seasonal
        
        # IMPLICATION with base=2000 (large) vs base=500 (small):
        # Large market (tier 0, base=2000):
        #   - amplitude_mult = 0.7
        #   - seasonal swing = ±21% (0.7 × 0.15 × 2)
        #   - absolute swing = ±420 bookings
        # Small market (tier 3, base=500):
        #   - amplitude_mult = 1.3
        #   - seasonal swing = ±39% (1.3 × 0.15 × 2)
        #   - absolute swing = ±195 bookings
        # Result: Large markets bigger absolute swings, small markets bigger % swings ✓
        size_tier_seasonal[tier_idx] = 1 + cfg['size_seasonal_amplitude'] * amplitude_mult * np.sin(2 * np.pi * t / 52 - np.pi/2 + phase_shift)
    

    # Generate data
    data = []
    for geo_idx in range(n_geos):
        base = geo_baseline_level[geo_idx]
        region = geo_region[geo_idx]
        size_tier = geo_size_tier[geo_idx]
        
        # Seasonality composition: Multiple layers with different weights
        # REASONING: Decompose seasonality into components at different levels
        #   - National (40%): All US geos share Christmas, Thanksgiving, summer
        #   - Regional (25%): West Coast vs Northeast differences (amplitude/timing)
        #   - Size tier (20%): Large diversified markets vs small specialized markets
        #   - Geo-specific (15%): Idiosyncratic local factors (one-off events, local industries)
        geo_seasonal = 1 + geo_seasonal_amplitude[geo_idx] * np.sin(2 * np.pi * t / 52 - np.pi/2 + np.random.uniform(-0.5, 0.5))
        
        seasonal = (0.40 * national_seasonal + 
                   0.25 * regional_seasonal[region] + 
                   0.20 * size_tier_seasonal[size_tier] + 
                   0.15 * geo_seasonal)
        
        # IMPLICATION: Geos in same region + same size tier will be highly correlated
        # This makes them natural controls/donors for each other in experiments
        # Example: SF and Seattle (same region, similar size) → great synthetic control match
        
        # Trend (geo-specific growth/decline)
        trend = (1 + geo_growth_rate[geo_idx]) ** (t / 52)
        
        # AR(1) noise (persistent shocks - good week followed by good week)
        ar_noise = np.zeros(n_weeks)
        ar_noise[0] = np.random.normal(0, cfg['ar_sigma'])
        for week in range(1, n_weeks):
            ar_noise[week] = cfg['ar_phi'] * ar_noise[week-1] + np.random.normal(0, cfg['ar_sigma'])
        ar_noise = 1 + ar_noise
        
        # Baseline bookings = base level × seasonal × trend × noise
        # The multiplication ensures larger geos have larger absolute swings
        baseline_bookings = base * seasonal * trend * ar_noise

        # demand proxies for MMM controls 
        demand_perfect = baseline_bookings.copy()  # for potential future use
        noise_good = np.random.normal(1.0, 0.08, n_weeks)
        demand_good = baseline_bookings * noise_good # mild noise
        demand_good = np.roll(demand_good, 1)  # lag by 1 week
        demand_good[0] = demand_good[1] # fix first week after roll

        noise_poor = np.random.normal(1.0, 0.30, n_weeks)
        wrong_lag = np.random.randint(0, 4)  # lag by 0-3 weeks
        demand_poor_lagged = np.roll(baseline_bookings, wrong_lag) 
        demand_poor_lagged[:wrong_lag] = demand_poor_lagged[wrong_lag]  # fix start after roll
        demand_poor = np.log1p(np.maximum(0, demand_poor_lagged * noise_poor)) * 50  # heavy noise + log transform


        # Market conditions (slow-moving exogenous control)
        market_trend = (1.02) ** (t / 52)  # 2% annual growth
        market_ar = np.zeros(n_weeks)
        market_ar[0] = np.random.normal(0, 0.05)
        for week in range(1, n_weeks):
            market_ar[week] = 0.8 * market_ar[week-1] + np.random.normal(0, 0.05)
        market_conditions = base * market_trend + market_ar * base * 0.1
        
        # Add rows
        for week_idx in range(n_weeks):
            data.append({
                'date': dates[week_idx],
                'geo': geo_idx,
                'region': region,
                'size_tier': size_tier,
                'population': int(geo_population[geo_idx]),
                'baseline_bookings': int(max(50, baseline_bookings[week_idx])),
                'market_conditions': market_conditions[week_idx],
                'demand_perfect': int(demand_perfect[week_idx]),
                'demand_good': demand_good[week_idx],
                'demand_poor': demand_poor[week_idx]
            })
    
    return pd.DataFrame(data)


def print_summary(df):
    """Print quick summary stats for validation."""
    print("="*70)
    print("BASELINE GEO DATA SUMMARY")
    print("="*70)
    print(f"Geos: {df['geo'].nunique()}, Weeks: {df['date'].nunique()}, Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nBaseline Bookings: mean={df['baseline_bookings'].mean():,.0f}, "
          f"range=[{df['baseline_bookings'].min():,}, {df['baseline_bookings'].max():,}]")
    print(f"Population: mean={df['population'].mean():,.0f}, "
          f"range=[{df['population'].min():,}, {df['population'].max():,}]")
    
    # Validate demand proxy quality
    print("\n" + "="*70)
    print("DEMAND PROXY QUALITY (correlation with baseline_bookings)")
    print("="*70)
    for proxy in ['demand_perfect', 'demand_good', 'demand_poor']:
        corr = df.groupby('geo').apply(
            lambda x: x['baseline_bookings'].corr(x[proxy])
        ).mean()
        print(f"{proxy:20s}: {corr:.3f}")
    
    print("\nExpected ranges:")
    print("  demand_perfect  : 1.000 (identical to baseline)")
    print("  demand_good     : 0.90-0.95 (small noise, 1-week lag)")
    print("  demand_poor     : 0.40-0.55 (high noise, wrong lag, log transform)")
    print("="*70)



if __name__ == "__main__":
    # Example usage
    df = generate_baseline_geo_data(n_geos=20, n_weeks=104)
    print_summary(df)