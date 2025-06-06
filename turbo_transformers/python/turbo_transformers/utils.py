# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
import contextlib

__all__ = [
    'pref_guard', 'set_num_threads', 'set_num_streams',
    'set_stderr_verbose_level', 'disable_perf', 'enable_perf',
    'reset_allocator_schema', 'bert_opt_mem_allocate_api',
    'print_results', 'get_gpu_mem_usage', 'set_task_to_stream_mapping',
    'clear_task_to_stream_mapping', 'get_stream_id_for_task'
]

set_num_threads = cxx.set_num_threads
set_num_streams = cxx.set_num_streams
set_stderr_verbose_level = cxx.set_stderr_verbose_level
set_task_to_stream_mapping = cxx.set_task_to_stream_mapping
clear_task_to_stream_mapping = cxx.clear_task_to_stream_mapping
get_stream_id_for_task = cxx.get_stream_id_for_task

disable_perf = cxx.disable_perf
enable_perf = cxx.enable_perf
print_results = cxx.print_results
get_gpu_mem_usage = cxx.get_gpu_mem_usage
reset_allocator_schema = cxx.reset_allocator_schema
bert_opt_mem_allocate_api = cxx.bert_opt_mem_allocate_api

@contextlib.contextmanager
def pref_guard(filename: str):
    cxx.enable_perf(filename)
    yield
    cxx.disable_perf()
