# Epic 6, Story 3: Token Counting & Cost Analysis

## Story Overview

**Epic**: LLM Provider Ecosystem Integration  
**Story ID**: 6.3  
**Priority**: Medium  
**Effort**: 3 Story Points  

## User Story

**As a** cost-conscious developer  
**I want** to track token usage and analyze costs across providers  
**So that** I can optimize spending and make informed provider decisions  

## Acceptance Criteria

- [ ] Accurate token counting for all supported providers
- [ ] Real-time cost calculation and tracking
- [ ] Cost comparison across different providers
- [ ] Usage analytics and spending trends
- [ ] Budget alerts and cost optimization recommendations

## TDD Requirements

- Write tests for token counting before implementing counting logic
- Test cost calculations before creating cost tracking
- Verify analytics accuracy before implementing reporting

## Definition of Done

- [ ] Token counting is accurate for all providers
- [ ] Cost calculations match provider pricing
- [ ] Real-time tracking works reliably
- [ ] Analytics provide actionable insights
- [ ] All tests pass in RED-GREEN-REFACTOR cycle
- [ ] Interactive cost dashboard is functional
- [ ] Budget management features work correctly
- [ ] Cost optimization recommendations are relevant

## Technical Implementation Notes

### Token Counting Components
```python
# Core token counting and cost analysis modules
from src.llm.factory import LLMFactory
from src.llm.providers import TokenCounter
from src.utils.cost_calculator import CostCalculator
from src.utils.analytics import UsageAnalytics
from src.utils.monitoring import SystemMonitor
from src.config.settings import ChunkingConfig

# Interactive widgets and visualization
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
```

### Token Counting Functions
```python
def create_token_counter():
    """Create universal token counter for all providers"""
    pass

def count_tokens_by_provider(text, provider_name, model_name):
    """Count tokens for specific provider and model"""
    pass

def estimate_tokens_before_request(text, provider_name, model_name):
    """Estimate token count before making API request"""
    pass

def track_actual_token_usage(request_data, response_data):
    """Track actual token usage from API responses"""
    pass

def validate_token_counting_accuracy(estimated, actual):
    """Validate accuracy of token count estimates"""
    pass
```

### Cost Calculation System
```python
def create_cost_calculator():
    """Create cost calculator with current provider pricing"""
    pass

def calculate_request_cost(provider_name, model_name, input_tokens, output_tokens):
    """Calculate cost for a specific request"""
    pass

def get_provider_pricing(provider_name, model_name):
    """Get current pricing for provider and model"""
    pass

def update_pricing_data():
    """Update pricing data from provider APIs or manual input"""
    pass

def calculate_monthly_projection(current_usage, days_elapsed):
    """Calculate projected monthly costs based on current usage"""
    pass
```

### Usage Analytics
```python
def create_usage_tracker():
    """Create comprehensive usage tracking system"""
    pass

def log_api_usage(provider_name, model_name, tokens_used, cost, timestamp):
    """Log API usage for analytics"""
    pass

def generate_usage_report(start_date, end_date, provider_filter=None):
    """Generate detailed usage report for specified period"""
    pass

def analyze_usage_patterns():
    """Analyze usage patterns and identify trends"""
    pass

def detect_usage_anomalies():
    """Detect unusual usage patterns or cost spikes"""
    pass
```

### Budget Management
```python
def create_budget_manager():
    """Create budget management and alerting system"""
    pass

def set_budget_limits(daily_limit, monthly_limit, provider_limits):
    """Set budget limits for different time periods and providers"""
    pass

def check_budget_status():
    """Check current budget status and remaining allowance"""
    pass

def trigger_budget_alerts(current_spending, budget_limits):
    """Trigger alerts when approaching budget limits"""
    pass

def recommend_cost_optimizations():
    """Recommend ways to optimize costs based on usage patterns"""
    pass
```

## Test Cases

### Test Case 1: Token Counting Accuracy
```python
def test_token_counting_accuracy():
    """Test accuracy of token counting across providers"""
    # RED: Write failing test for token counting
    # GREEN: Implement accurate token counting
    # REFACTOR: Optimize counting performance
    pass
```

### Test Case 2: Cost Calculation Precision
```python
def test_cost_calculation_precision():
    """Test precision of cost calculations"""
    # RED: Write failing test for cost calculations
    # GREEN: Implement precise cost calculation
    # REFACTOR: Improve calculation efficiency
    pass
```

### Test Case 3: Real-time Tracking
```python
def test_realtime_tracking():
    """Test real-time usage and cost tracking"""
    # RED: Write failing test for real-time tracking
    # GREEN: Implement real-time tracking
    # REFACTOR: Optimize tracking performance
    pass
```

### Test Case 4: Budget Management
```python
def test_budget_management():
    """Test budget limits and alerting system"""
    # RED: Write failing test for budget management
    # GREEN: Implement budget management
    # REFACTOR: Improve budget accuracy
    pass
```

### Test Case 5: Analytics Accuracy
```python
def test_analytics_accuracy():
    """Test accuracy of usage analytics and reporting"""
    # RED: Write failing test for analytics
    # GREEN: Implement analytics system
    # REFACTOR: Enhance analytics insights
    pass
```

## Interactive Features

### Cost Dashboard
- Real-time cost tracking display
- Provider cost comparison charts
- Daily/weekly/monthly spending trends
- Budget status indicators
- Cost per request metrics

### Token Usage Analytics
- Token usage by provider and model
- Input vs output token distribution
- Token efficiency metrics
- Usage pattern visualization
- Peak usage identification

### Budget Management Interface
- Budget limit configuration
- Spending alerts setup
- Budget utilization progress bars
- Cost projection calculations
- Budget optimization suggestions

### Provider Cost Comparison
- Side-by-side cost comparison
- Cost per token analysis
- Model pricing comparison
- ROI analysis by provider
- Cost-effectiveness rankings

## Provider Pricing Integration

### OpenAI Pricing
```python
openai_pricing = {
    'gpt-4': {
        'input': 0.03,   # per 1K tokens
        'output': 0.06   # per 1K tokens
    },
    'gpt-3.5-turbo': {
        'input': 0.0015,
        'output': 0.002
    },
    'text-embedding-ada-002': {
        'input': 0.0001,
        'output': 0.0
    }
}
```

### Anthropic Pricing
```python
anthropic_pricing = {
    'claude-3-sonnet': {
        'input': 0.015,
        'output': 0.075
    },
    'claude-3-haiku': {
        'input': 0.00025,
        'output': 0.00125
    }
}
```

### Google Pricing
```python
google_pricing = {
    'gemini-pro': {
        'input': 0.0005,
        'output': 0.0015
    },
    'text-bison': {
        'input': 0.001,
        'output': 0.001
    }
}
```

## Analytics Visualizations

### Cost Trend Analysis
```python
def create_cost_trend_chart(usage_data):
    """Create interactive cost trend visualization"""
    fig = go.Figure()
    
    # Add cost trends by provider
    for provider in usage_data['provider'].unique():
        provider_data = usage_data[usage_data['provider'] == provider]
        fig.add_trace(go.Scatter(
            x=provider_data['date'],
            y=provider_data['cost'],
            mode='lines+markers',
            name=provider,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Cost Trends by Provider',
        xaxis_title='Date',
        yaxis_title='Cost ($)',
        hovermode='x unified'
    )
    
    return fig
```

### Token Usage Distribution
```python
def create_token_distribution_chart(token_data):
    """Create token usage distribution visualization"""
    fig = px.pie(
        token_data,
        values='tokens',
        names='provider',
        title='Token Usage Distribution by Provider'
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig
```

### Cost Efficiency Analysis
```python
def create_efficiency_analysis(efficiency_data):
    """Create cost efficiency analysis visualization"""
    fig = px.scatter(
        efficiency_data,
        x='tokens_per_dollar',
        y='quality_score',
        size='usage_volume',
        color='provider',
        title='Provider Cost Efficiency Analysis',
        labels={
            'tokens_per_dollar': 'Tokens per Dollar',
            'quality_score': 'Quality Score',
            'usage_volume': 'Usage Volume'
        }
    )
    
    return fig
```

## Cost Optimization Strategies

### Model Selection Optimization
```python
def recommend_optimal_model(task_requirements, budget_constraints):
    """Recommend optimal model based on requirements and budget"""
    recommendations = {
        'high_quality_tasks': 'gpt-4 for critical tasks, claude-3-sonnet for analysis',
        'bulk_processing': 'gpt-3.5-turbo for volume, claude-3-haiku for speed',
        'embeddings': 'text-embedding-ada-002 for OpenAI, jina for alternatives',
        'cost_sensitive': 'claude-3-haiku, gpt-3.5-turbo, google models'
    }
    return recommendations
```

### Usage Pattern Optimization
```python
def analyze_usage_efficiency(usage_history):
    """Analyze usage patterns for efficiency improvements"""
    optimizations = {
        'batch_processing': 'Combine small requests into batches',
        'caching': 'Implement caching for repeated queries',
        'model_switching': 'Use cheaper models for simple tasks',
        'prompt_optimization': 'Optimize prompts to reduce token usage'
    }
    return optimizations
```

## Budget Alert System

### Alert Configuration
```python
alert_thresholds = {
    'daily_warning': 0.8,    # 80% of daily budget
    'daily_critical': 0.95,  # 95% of daily budget
    'monthly_warning': 0.75, # 75% of monthly budget
    'monthly_critical': 0.9, # 90% of monthly budget
    'spike_detection': 2.0   # 2x normal usage
}
```

### Alert Actions
```python
def handle_budget_alert(alert_type, current_usage, budget_limit):
    """Handle different types of budget alerts"""
    actions = {
        'warning': 'Send notification, suggest optimizations',
        'critical': 'Send urgent notification, recommend immediate action',
        'exceeded': 'Block new requests, require manual override',
        'spike': 'Investigate unusual usage, check for errors'
    }
    return actions[alert_type]
```

## Success Metrics

- **Token Counting Accuracy**: >99% accuracy compared to provider billing
- **Cost Tracking Precision**: <1% variance from actual costs
- **Real-time Performance**: Cost updates within 1 second
- **Budget Compliance**: 100% alert accuracy for budget thresholds
- **Cost Optimization**: 15-30% cost reduction through recommendations

## Dependencies

- Epic 6, Story 1: Multi-Provider Integration Demo
- Epic 6, Story 2: Dynamic Provider Switching
- TokenCounter, CostCalculator components
- Provider API access for pricing data

## Related Stories

- Epic 6, Story 4: Provider Performance Benchmarking
- Epic 7, Story 2: Intelligent Caching System
- Epic 4, Story 1: Performance Monitoring Dashboard
- Epic 9, Story 3: Production Monitoring Simulation

---

**Created**: 2024-12-19  
**Last Updated**: 2024-12-19  
**Status**: Ready for Development  
**Assigned**: TBD