#include "ztree_wrapper.hpp"

size_t key_size_ = 0;
size_t pool_size_ = ((size_t)(1024 * 1024 * 16) * 1000);
const char *pool_path_;

extern "C" tree_api* create_tree(const tree_options_t& opt)
{
	auto path_ptr = new std::string(opt.pool_path);
    pool_path_ = (*path_ptr).c_str();
    return new ztree_wrapper(pool_path_);
}