#pragma once
#include<vector>
#include<map>
#include "tensor.h"
#include "sparse_tensor.h"
#include "turbo_transformers/layers/kernels/sparse_mat_mul.h"
#include "3rd/seqmm/seqmm/types.h"

#define SP_SHADOW

#define SHADOW_PERF_DEBUG 0

namespace turbo_transformers {

// Make sure that the types are consistant with those in python 
enum PET_TYPEs {
    MASK_BERT,
    DIFF_PRUNING,
    BITFIT,
    ADAPTERS,
    STANDARD,
    LORA
};

namespace core {

class PETLayerManager{
public:
    PETLayerManager() {
        total_tasks = 0;
    }

    ~PETLayerManager() {
#ifdef SP_SHADOW 
        destroy_sparse_matmuls();
#endif
    }
    void load_new_task(int pet_type,
            bool has_layer_norm, 
            core::Tensor * task_mask, 
            core::Tensor * task_diff,
            core::Tensor * task_bias,
            core::Tensor * task_layer_norm_weight,
            core::Tensor * task_layer_norm_bias,
            core::Tensor * down_scale_w,
            core::Tensor * down_scale_b,
            core::Tensor * up_scale_w,
            core::Tensor * up_scale_b
             ){
        switch (pet_type)
        {
        case MASK_BERT:
            load_new_maskbert_task(
                    has_layer_norm, 
                    *task_mask, 
                    *task_layer_norm_weight, 
                    *task_layer_norm_bias);
#ifdef SP_SHADOW 
            create_sparse_matmul(total_tasks);
#endif
            break;
        case DIFF_PRUNING:
            load_new_diff_task(
                    has_layer_norm,
                    *task_diff,
                    *task_bias,
                    *task_layer_norm_weight,
                    *task_layer_norm_bias);
#ifdef SP_SHADOW 
            create_sparse_matmul(total_tasks);
#endif
            break;
        case BITFIT:
            load_new_bitfit_task(
                    has_layer_norm,
                    *task_bias,
                    *task_layer_norm_weight,
                    *task_layer_norm_bias);
            break;
        case ADAPTERS:
            if (down_scale_w != nullptr && down_scale_b != nullptr && 
                up_scale_w != nullptr && up_scale_b != nullptr) {
                static core::Tensor dummy_ln_w_adapter(nullptr); // Initialize with nullptr
                static core::Tensor dummy_ln_b_adapter(nullptr); // Initialize with nullptr

                if (has_layer_norm && (!task_layer_norm_weight || !task_layer_norm_bias)) {
                     std::cerr << "Warning: ADAPTERS task specified has_layer_norm=true, but LayerNorm tensor pointers are null. Using empty tensors for safety." << std::endl;
                }

                core::Tensor& ref_ln_w = (has_layer_norm && task_layer_norm_weight) ? *task_layer_norm_weight : dummy_ln_w_adapter;
                core::Tensor& ref_ln_b = (has_layer_norm && task_layer_norm_bias)   ? *task_layer_norm_bias   : dummy_ln_b_adapter;

                load_new_adapter_task(
                    has_layer_norm,
                    *down_scale_w,
                    *down_scale_b,
                    *up_scale_w,
                    *up_scale_b,
                    ref_ln_w,
                    ref_ln_b
                );
            }
            // If primary adapter pointers are null (e.g., from BertIntermediate), this block is skipped.
            break;
        case STANDARD:
            break;
        case LORA:
            if (down_scale_w != nullptr && down_scale_b != nullptr && 
                up_scale_w != nullptr && up_scale_b != nullptr) { // Check primary pointers are not null
                static core::Tensor dummy_ln_w_lora(nullptr); // Initialize with nullptr
                static core::Tensor dummy_ln_b_lora(nullptr); // Initialize with nullptr
                
                if (has_layer_norm && (!task_layer_norm_weight || !task_layer_norm_bias)) {
                     std::cerr << "Warning: LORA task specified has_layer_norm=true, but LayerNorm tensor pointers are null. Using empty tensors for safety." << std::endl;
                }

                core::Tensor& ref_ln_w = (has_layer_norm && task_layer_norm_weight) ? *task_layer_norm_weight : dummy_ln_w_lora;
                core::Tensor& ref_ln_b = (has_layer_norm && task_layer_norm_bias)   ? *task_layer_norm_bias   : dummy_ln_b_lora;

                load_new_lora_task(
                    has_layer_norm,
                    *down_scale_w,
                    *down_scale_b,
                    *up_scale_w,
                    *up_scale_b,
                    ref_ln_w,
                    ref_ln_b
                );
            }
            // If primary lora pointers are null (e.g., from BertIntermediate), this block is skipped.
            break;
        default:
            break;
        }

        loaded_pets_types_[total_tasks] = pet_type;
        total_tasks ++;
    }

    int get_pet_type(int task_id) const{
        auto iter = loaded_pets_types_.find(task_id);
        assert(iter!=loaded_pets_types_.end());
        return iter->second;
    }
    
    // get shadows
    const core::Tensor& get_maskbert_shadow(const int task_id) const{
        auto iter = task_masks_.find(task_id);
        assert(iter!=task_masks_.end());
        return iter->second;
    }

    const core::Tensor& get_diff_shadow(const int task_id) const{
        auto iter = task_diffs_.find(task_id);
        assert(iter!=task_diffs_.end());
        return iter->second;
    }

    core::SparseTensor * get_sp_diff_shadow(const int task_id) const{
        auto iter = sp_task_diffs_.find(task_id);
        assert(iter!=sp_task_diffs_.end());
        return iter->second;
    }

    core::SparseTensor * get_sp_mask_shadow(const int task_id) const{
        auto iter = sp_task_masks_.find(task_id);
        assert(iter!=sp_task_masks_.end());
        return iter->second;
    }

    const core::Tensor& get_bias(const int task_id) const{
        if(task_id == -1){
            return *shared_bias;
        }
        else{
            auto iter = task_biases_.find(task_id);
            assert(iter!=task_biases_.end());
            return iter->second;
        }

    }

    const core::Tensor& get_layer_norm_bias(const int task_id) const{
        if(task_id == -1){
            return *shared_layer_norm_bias;
        }
        else{
            auto iter = task_norm_biases_.find(task_id);
            assert(iter!=task_norm_biases_.end());
            return iter->second;
        }

    }

    const core::Tensor& get_layer_norm_weight(const int task_id) const{
        if(task_id == -1){
            return *shared_layer_norm_weight;
        }
        else{
            auto iter = task_norm_weights_.find(task_id);
            assert(iter!=task_norm_weights_.end());
            return iter->second;
        }
    }

    const core::Tensor& get_adapter_params(const int task_id, bool is_down_scale, bool is_weight ) const {
        if(is_down_scale){
            if(is_weight){
                auto iter = down_scale_weights_.find(task_id);
                assert(iter!=down_scale_weights_.end());
                return iter->second;
            }
            else{
                auto iter = down_scale_biases_.find(task_id);
                assert(iter!=down_scale_biases_.end());
                return iter->second;
            }
        }else{
            if(is_weight){
                auto iter = up_scale_weights_.find(task_id);
                assert(iter!=up_scale_weights_.end());
                return iter->second;
            }
            else{
                auto iter = up_scale_biases_.find(task_id);
                assert(iter!=up_scale_biases_.end());
                return iter->second;
            } 
        }
    }

    const core::Tensor& get_lora_A(const int task_id) const {
        auto iter = lora_A_.find(task_id);
        assert(iter != lora_A_.end());
        return iter->second;
    }

    const core::Tensor& get_lora_A_bias_zero(const int task_id) const {
        auto iter = lora_A_biases_zero_.find(task_id);
        assert(iter != lora_A_biases_zero_.end());
        return iter->second;
    }

    const core::Tensor& get_lora_B(const int task_id) const {
        auto iter = lora_B_.find(task_id);
        assert(iter != lora_B_.end());
        return iter->second;
    }

    const core::Tensor& get_lora_B_bias_zero(const int task_id) const {
        auto iter = lora_B_biases_zero_.find(task_id);
        assert(iter != lora_B_biases_zero_.end());
        return iter->second;
    }

    std::shared_ptr<layers::kernels::SparseMatMulCsr> get_sparse_matmul(const int task_id) const {
        return sparse_matmul_map_.at(task_id);
    }

public:
    core::Tensor * shared_weight;
    core::Tensor * shared_bias;
    core::Tensor * shared_layer_norm_bias;
    core::Tensor * shared_layer_norm_weight;

private:
    int total_tasks;
    // map: task_id->PET_TYPES
    std::map<int, int> loaded_pets_types_;
    
    // for dense baseline
    std::map<int, core::Tensor> task_masks_;
    std::map<int, core::Tensor> task_diffs_;
    
    // sparse tensor 
    std::map<int, core::SparseTensor *> sp_task_masks_;
    std::map<int, core::SparseTensor *> sp_task_diffs_;
    
    // bias and norm parameters are always dense
    std::map<int, core::Tensor> task_biases_;
    std::map<int, core::Tensor> task_norm_weights_;
    std::map<int, core::Tensor> task_norm_biases_;

    // upscale and downscale parameters for adapters
    std::map<int, core::Tensor> up_scale_weights_;
    std::map<int, core::Tensor> up_scale_biases_;
    std::map<int, core::Tensor> down_scale_weights_;
    std::map<int, core::Tensor> down_scale_biases_;

    // Parameters for LoRA (A and B matrices, and their corresponding zero biases)
    std::map<int, core::Tensor> lora_A_;
    std::map<int, core::Tensor> lora_A_biases_zero_;
    std::map<int, core::Tensor> lora_B_;
    std::map<int, core::Tensor> lora_B_biases_zero_;

    std::map<int, std::shared_ptr<layers::kernels::SparseMatMulCsr>> sparse_matmul_map_;


/* For loading PET parameters */

    void load_new_maskbert_task(bool has_layer_norm,
                                core::Tensor & task_mask,
                                core::Tensor & task_layer_norm_weight,
                                core::Tensor & task_layer_norm_bias){
        
#ifdef SP_SHADOW     
        SparseTensor* sp_mask_ptr = new core::SparseTensorCsr<float>(task_mask);
        sp_mask_ptr->Dense2Sparse();
        sp_task_masks_.insert(std::make_pair(total_tasks, sp_mask_ptr));
#else
        task_masks_.insert(std::make_pair(total_tasks, std::move(task_mask)));
#endif
        if(has_layer_norm){
            task_norm_weights_.insert(std::make_pair(total_tasks, std::move(task_layer_norm_weight)));
            task_norm_biases_.insert(std::make_pair(total_tasks, std::move(task_layer_norm_bias)));
        }
    }
    void load_new_diff_task(bool has_layer_norm,
                            core::Tensor & task_diff,
                            core::Tensor & task_bias,
                            core::Tensor & task_layer_norm_weight,
                            core::Tensor & task_layer_norm_bias){
        
#ifdef SP_SHADOW   
        SparseTensor* sp_diff_ptr = new core::SparseTensorCsr<float>(task_diff);
        sp_diff_ptr->Dense2Sparse();
        sp_task_diffs_.insert(std::make_pair(total_tasks, sp_diff_ptr));
#else
        task_diffs_.insert(std::make_pair(total_tasks, std::move(task_diff)));
#endif
        task_biases_.insert(std::make_pair(total_tasks, std::move(task_bias)));
        if(has_layer_norm){
            task_norm_weights_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_weight)));
            task_norm_biases_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_bias)));
        }
    }
    void load_new_bitfit_task(bool has_layer_norm,
                            core::Tensor & task_bias,
                            core::Tensor & task_layer_norm_weight,
                            core::Tensor & task_layer_norm_bias){
        task_biases_.insert(std::make_pair(total_tasks, std::move(task_bias)));
        if(has_layer_norm){
            task_norm_weights_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_weight)));
            task_norm_biases_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_bias)));
        }
    }

    void load_new_adapter_task(
            bool has_layer_norm,
            core::Tensor & down_scale_w,
            core::Tensor & down_scale_b,
            core::Tensor & up_scale_w,
            core::Tensor & up_scale_b,
            core::Tensor & task_layer_norm_weight,
            core::Tensor & task_layer_norm_bias
             ){
   
            down_scale_weights_.insert(std::make_pair(total_tasks,std::move(down_scale_w)));
            down_scale_biases_.insert(std::make_pair(total_tasks,std::move(down_scale_b)));
            up_scale_weights_.insert(std::make_pair(total_tasks,std::move(up_scale_w))); 
            up_scale_biases_.insert(std::make_pair(total_tasks,std::move(up_scale_b))); 

            if(has_layer_norm){
                task_norm_weights_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_weight)));
                task_norm_biases_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_bias)));
        }
    }

    void load_new_lora_task(
            bool has_layer_norm,
            core::Tensor & lora_a,
            core::Tensor & lora_a_b_zero,
            core::Tensor & lora_b,
            core::Tensor & lora_b_b_zero,
            core::Tensor & task_layer_norm_weight,
            core::Tensor & task_layer_norm_bias
             ){
   
            lora_A_.insert(std::make_pair(total_tasks,std::move(lora_a)));
            lora_A_biases_zero_.insert(std::make_pair(total_tasks,std::move(lora_a_b_zero)));
            lora_B_.insert(std::make_pair(total_tasks,std::move(lora_b)));
            lora_B_biases_zero_.insert(std::make_pair(total_tasks,std::move(lora_b_b_zero))); 

            if(has_layer_norm){
                task_norm_weights_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_weight)));
                task_norm_biases_.insert(std::make_pair(total_tasks,std::move(task_layer_norm_bias)));
        }
    }

/* For sparse operations */
    void create_sparse_matmul(const int task_id) {
        // FIXME: determine sparse operator type according to sparse pattern of 
        // task weight.
        int stream_id = task_id % CUDADeviceContext::num_streams;
        //std::shared_ptr<layers::kernels::SparseMatMul> sparse_matmul_ptr = 
        //    layers::kernels::SparseMatMul::SparseMatMulFactory(seqmm::SparseFormat::kFmtCSR);
        std::shared_ptr<layers::kernels::SparseMatMulCsr> sparse_matmul_ptr = 
        layers::kernels::SparseMatMulCsr::GetInstance(stream_id, seqmm::SparseFormat::kFmtCSR);
        
        sparse_matmul_map_.insert(std::make_pair(task_id, sparse_matmul_ptr));
    }

    void destroy_sparse_matmuls() {
        std::vector<int> stream_id_vec;
        for (auto const& _ : sparse_matmul_map_) {
#ifdef PRINT_DEBUG_INFO
            std::cout << "Destroy SparseMatmul for shadow task #" 
                    << loaded_pets_types_[_.first] << "..." << std::endl;
#endif     
#ifdef PRINT_DEBUG_INFO
            std::cout << "Done." << std::endl;
#endif
        }
    }
};
}
}
