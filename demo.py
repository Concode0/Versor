# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

"""Interactive Streamlit Demo for Versor.

Provides a graphical interface to generate distorted manifold data,
train a Geometric Algebra network live, and visualize the unbending process.
"""

import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from layers.projection import BladeSelector

# Setup Page
st.set_page_config(page_title="Versor: Geometric Algebra Demo", layout="wide")

st.title("🌀 Versor: Geometric Algebra Neural Engine")
st.markdown("""
This interactive demo visualizes **Geometric Algebra (GA)** transformations.
Versor uses **Multivectors** and **Rotors** to manipulate high-dimensional geometry.
""")

# Sidebar Controls
st.sidebar.header("Configuration")

# 1. Data Generation Params
st.sidebar.subheader("Data Generation")
samples = st.sidebar.slider("Number of Samples", 100, 10000, 500)
distortion = st.sidebar.slider("Distortion Factor (Z = k*XY)", 0.0, 5.0, 0.5)

# 2. Training Params
st.sidebar.subheader("Training Parameters")
train_epochs = st.sidebar.slider("Epochs", 10, 1000, 50)
lr = st.sidebar.number_input("Learning Rate", 0.001, 0.1, 0.05)

start_align = st.sidebar.button("🚀 Align Manifold (Train)", type="primary")

def plot_3d_manifold(data, title, color_data=None):
    """Generates an interactive 3D scatter plot using Plotly.

    Args:
        data (torch.Tensor): Point cloud data [N, Dim].
        title (str): Plot title.
        color_data (numpy.ndarray, optional): Array for coloring points. Defaults to Z-value.

    Returns:
        plotly.graph_objects.Figure: The interactive figure.
    """
    # Ensure data is [N, Dim]
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
        if data.ndim == 3: # [N, 1, D] -> [N, D]
            data = data.squeeze(1)
        
    x = data[:, 1].numpy()
    y = data[:, 2].numpy()
    z = data[:, 4].numpy()
    
    if color_data is None:
        color = z 
    else:
        color = color_data
        
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=color,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Z-Value")
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='e1 (X)',
            yaxis_title='e2 (Y)',
            zaxis_title='e3 (Z)',
            aspectmode='data' 
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# --- Main App Logic (Manifold Unbending) ---

# 1. Generate Data
algebra = CliffordAlgebra(3, 0, device='cpu')
t = torch.linspace(0, 2*np.pi, samples)
x = torch.sin(t)
y = torch.sin(t) * torch.cos(t)
z = distortion * x * y

# Input must be [Batch, Channels, Dim] for RotorLayer
data = torch.zeros(samples, 1, algebra.dim)
data[:, 0, 1] = x
data[:, 0, 2] = y
data[:, 0, 4] = z

original_z_colors = z.numpy()

# 2. State Management
param_hash = f"{samples}_{distortion}"

if 'last_params' not in st.session_state:
    st.session_state.last_params = param_hash
    st.session_state.trained_output = None
    
if st.session_state.last_params != param_hash:
    st.session_state.trained_output = None
    st.session_state.last_params = param_hash

# 3. Layout
col1, col2 = st.columns(2)

with col1:
    st.info("Input: Distorted Manifold")
    st.plotly_chart(plot_3d_manifold(data, "Original Input", color_data=original_z_colors), key="plot_input")

output_placeholder = col2.empty()

# 4. Render Logic
if start_align:
    # --- TRAINING MODE ---
    class ManifoldNet(torch.nn.Module):
        """Simple rotor-based network for the demo."""
        def __init__(self, alg):
            super().__init__()
            self.rotor = RotorLayer(alg, channels=1)
            self.selector = BladeSelector(alg, channels=1)
            
        def forward(self, x):
            x_rot = self.rotor(x)
            return self.selector(x_rot)

    model = ManifoldNet(algebra)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Progress indicators in main area
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    out = None
    for epoch in range(train_epochs):
        optimizer.zero_grad()
        out = model(data) # [S, 1, D]
        
        # Loss: Minimize energy in e3 (z-axis, index 4)
        loss = (out[:, 0, 4]**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == train_epochs - 1:
            progress_bar.progress((epoch + 1) / train_epochs)
            status_text.text(f"Epoch {epoch+1}/{train_epochs} | Loss (Z-Energy): {loss.item():.6f}")
            
            with output_placeholder.container():
                st.warning(f"Training in progress... (Epoch {epoch})")
                st.plotly_chart(
                    plot_3d_manifold(out, f"Unbending... Loss: {loss.item():.4f}", color_data=original_z_colors), 
                    width='stretch'
                )
    
    # Save final result strictly
    st.session_state.trained_output = out.detach().clone()
    status_text.empty()
    progress_bar.empty()
    st.success("Training Complete!")
    
    # Immediate final render
    with output_placeholder.container():
        st.success("Unbent Manifold (Result)")
        st.plotly_chart(plot_3d_manifold(st.session_state.trained_output, "Final Result", color_data=original_z_colors), key="plot_final")

else:
    # --- STATIC/IDLE MODE ---
    with output_placeholder.container():
        if st.session_state.trained_output is not None:
            st.success("Unbent Manifold (Result)")
            st.plotly_chart(plot_3d_manifold(st.session_state.trained_output, "Final Result", color_data=original_z_colors), key="plot_result_static")
        else:
            st.info("Current State (Same as Input)")
            st.plotly_chart(plot_3d_manifold(data, "Waiting to Train...", color_data=original_z_colors), key="plot_waiting")

st.markdown("---")
st.markdown("Powered by **Versor** | [GitHub Repository](https://github.com/Concode0/Versor)")