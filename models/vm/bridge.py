# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Frozen LLM prefix bridge for the Geometric Turing Machine.

Runs the first `prefix_layers` transformer blocks of a pretrained GPT-2 model
and returns the intermediate hidden states (no ln_f). All parameters are frozen.
"""

import torch
import torch.nn as nn


class LLMBridge(nn.Module):
    """Frozen GPT-2 prefix bridge.

    Keeps the full GPT-2 model for state_dict compatibility, but only
    runs the first `prefix_layers` transformer blocks during forward.
    """

    def __init__(self, model_name='gpt2', prefix_layers=4):
        super().__init__()
        from transformers import GPT2Model
        self.model = GPT2Model.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size
        self.prefix_layers = prefix_layers
        # Freeze all
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use hidden state after prefix_layers (index = prefix_layers
            # because index 0 is embedding output).
            # No ln_f — this is an intermediate representation, not final.
            hidden = out.hidden_states[self.prefix_layers]
        return hidden
