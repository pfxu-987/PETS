// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "turbo_transformers/layers/bert_attention.h"

#include <unordered_map>

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"

namespace turbo_transformers {
namespace layers {

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output, core::Tensor* attn,
                               bool is_trans_weight,
                               
                               core::Tensor *task_ids,
                               core::Tensor *n_samples,
                               core::Tensor *minibatch_lens
                               ) const {
  std::unordered_map<std::string, core::Tensor*> dummy{};
  core::Tensor* attn_ptr;
  if (attn == nullptr) {
    attn_ptr = new core::Tensor(nullptr);
  } else {
    attn_ptr = attn;
  }
  MultiHeadedAttention::operator()(
      input_tensor, input_tensor, input_tensor, attention_mask, "self", output,
      attn_ptr, dummy, false /* pre_layernorm */, true /* post_layernorm */,
      false /* post_add_input */, is_trans_weight /* is_trans_weight */ , task_ids, n_samples, minibatch_lens);
  if (attn == nullptr) {
    delete attn_ptr;
  }
}

void BertAttention::EnforceShapeAndType() const {
  MultiHeadedAttention::EnforceShapeAndType();
}

void BertAttention::load_new_task(  
    int pet_type_int,
    core::Tensor& qkv_weight_mask_tensor,
    core::Tensor& qkv_weight_diff_tensor,
    core::Tensor& qkv_bias_tensor,
    core::Tensor& output_weight_mask_tensor,
    core::Tensor& output_weight_diff_tensor,
    core::Tensor& output_bias_tensor,
    core::Tensor& output_layerNorm_weight_tensor,
    core::Tensor& output_layerNorm_bias_tensor,
    core::Tensor& down_scale_w_tensor,
    core::Tensor& down_scale_b_tensor,
    core::Tensor& up_scale_w_tensor,
    core::Tensor& up_scale_b_tensor
) {
    PET_TYPEs current_pet_type_for_output = static_cast<PET_TYPEs>(pet_type_int);
    PET_TYPEs current_pet_type_for_qkv = static_cast<PET_TYPEs>(pet_type_int);

    if (current_pet_type_for_qkv == ADAPTERS || current_pet_type_for_qkv == LORA) {
        current_pet_type_for_qkv = STANDARD;
    }

    // For QKV, PETs like Adapters/LoRA (if they were to apply here) would not typically add their own LayerNorm
    // MaskBERT doesn't have bias/LN for QKV.
    // DiffPruning and BitFit might influence bias for QKV, but not typically a separate LN.
    // So, has_layer_norm for qkv_manager_ is generally false for these specific params.
    qkv_manager.load_new_task(
        current_pet_type_for_qkv, // Use modified type for QKV manager
        false, // has_layer_norm is false for QKV-specific PET params here
        &qkv_weight_mask_tensor, 
        &qkv_weight_diff_tensor, 
        &qkv_bias_tensor, 
        nullptr, // task_layer_norm_weight for QKV
        nullptr, // task_layer_norm_bias for QKV
        nullptr, // down_scale_w for QKV (not used by ADAPTERS/LORA here)
        nullptr, // down_scale_b for QKV
        nullptr, // up_scale_w for QKV
        nullptr  // up_scale_b for QKV
    );

    // For the output projection part (where Adapters/LoRA apply their projections and LN)
    bool has_ln_for_output_pet = true; // Default assumption
    if (current_pet_type_for_output == MASK_BERT) { // MASK_BERT doesn't have its own LN parameters for output
        has_ln_for_output_pet = false;
    }
    // Other PET types (DiffPruning, BitFit, Adapters, LoRA) will use the provided out_ln_w/b for their task-specific LN

    output_manager.load_new_task(
        current_pet_type_for_output, // Use original pet_type for output manager
        has_ln_for_output_pet, 
        &output_weight_mask_tensor, 
        &output_weight_diff_tensor, 
        &output_bias_tensor, 
        &output_layerNorm_weight_tensor, 
        &output_layerNorm_bias_tensor,
        &down_scale_w_tensor, // These are for Adapter/LoRA on the output dense layer
        &down_scale_b_tensor,
        &up_scale_w_tensor,
        &up_scale_b_tensor
    );
}

BertAttention::~BertAttention() {
}

}  // namespace layers
}  // namespace turbo_transformers
