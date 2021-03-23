args = {
    "device" : "cuda:0",
    
    "total_epoch" : 40,
    "mile_stones" : [10, 30],
    "lr" : 0.1,
    "optim" : 'sgd',
    
    "attn_type" : None,    # use CBAM
    "diagonal_input" : True,  
    "use_cag" : [True] *4,  # use CAG
    
    "model_save_dir" : "../final_model/",
    "img_info_path" : "../img_detail/detail.csv",
    "data_path" : "../sim_data/euc_100.npy"
    
}