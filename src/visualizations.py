import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_training_convergence(protocol_km=15):
    try:
        df = pd.read_csv(f'data/processed/training_history_{protocol_km}km.csv')
    except Exception:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Train_Loss'], mode='lines', name='Train Loss', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Val_Loss'], mode='lines', name='Validation Loss', line=dict(color='fuchsia')))
    
    fig.update_layout(title="MPINN Convergence (Huber Loss & Cosine Annealing)",
                      xaxis_title="Epoch", yaxis_title="Huber Loss (Robust to Rogue Outliers)",
                      template="plotly_dark", height=400)
    return fig

def plot_spatial_error(protocol_km=15):
    try:
        err_df = pd.read_csv(f'data/processed/spatial_errors_{protocol_km}km.csv')
        if 'Actual_E_ms' in err_df.columns:
            act_e = 'Actual_E_ms'
            act_n = 'Actual_N_ms'
            pred_e = 'Pred_E_ms'
            pred_n = 'Pred_N_ms'
        else:
            act_e = 'Actual_E'
            act_n = 'Actual_N'
            pred_e = 'Pred_E'
            pred_n = 'Pred_N'
    except Exception:
        return None
        
    fig = go.Figure()
    
    # Draw connecting residual lines
    for _, row in err_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[act_e], row[pred_e]],
            y=[row[act_n], row[pred_n]],
            mode='lines', line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            showlegend=False
        ))
        
    # Draw Actuals
    fig.add_trace(go.Scatter(
        x=err_df[act_e], y=err_df[act_n], 
        mode='markers', name='Actual', marker=dict(color='lime', size=6)
    ))
    
    # Draw Predicts
    fig.add_trace(go.Scatter(
        x=err_df[pred_e], y=err_df[pred_n], 
        mode='markers', name='Predicted (MPINN)', marker=dict(color='fuchsia', size=6, symbol='x')
    ))

    fig.update_layout(title="Spatial Residuals (Actual vs MPINN Vector)",
                      xaxis_title="Kinematic E", yaxis_title="Kinematic N",
                      template="plotly_dark", height=400)
    return fig

def plot_error_distribution(protocol_km=15):
    try:
        err_df = pd.read_csv(f'data/processed/spatial_errors_{protocol_km}km.csv')
        err_col = 'Error_KM_24h' if 'Error_KM_24h' in err_df.columns else 'Error_KM'
    except Exception:
        return None
        
    fig = px.histogram(err_df, x=err_col, nbins=30, 
                       title="Prediction Error Distribution (Bootstrapped Confidence)",
                       labels={err_col: 'Absolute Spatial Error (km)'},
                       color_discrete_sequence=['cyan'])
    fig.update_layout(template="plotly_dark", yaxis_title="Node Count", height=400)
    
    # Add vertical line for mean error
    mean_err = err_df[err_col].mean()
    fig.add_vline(x=mean_err, line_dash="dash", line_color="fuchsia", 
                  annotation_text=f"Mean Error: {mean_err:.2f}km")
                  
    return fig

def plot_forensic_lift(protocol_km=15):
    try:
        rules = pd.read_csv(f'data/processed/forensic_rules_{protocol_km}km.csv')
    except Exception:
        return None
        
    # Filter rules Lift > 1.5
    rules = rules[rules['lift'] > 1.5].copy()
    if rules.empty:
        return None
        
    # Format labels cleanly by stripping set syntax
    def clean_set(s):
        return str(s).replace("frozenset({", "").replace("})", "").replace("'", "")
    
    rules['Rule'] = rules['antecedents'].apply(clean_set) + " ➔ " + rules['consequents'].apply(clean_set)
    
    # Top 5 by Lift
    top_5 = rules.nlargest(5, 'lift')
    top_5 = top_5.sort_values(by='lift', ascending=True) # Ascending for horizontal bar orientation
    
    fig = px.bar(top_5, x='lift', y='Rule', orientation='h', 
                 title="Top 5 Forensic Signatures (Lift > 1.5)",
                 color='lift', color_continuous_scale=px.colors.sequential.Plasma,
                 labels={'lift': 'Information Lift Score'})
    fig.update_layout(template="plotly_dark", height=400)
    return fig
