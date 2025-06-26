# Get KLs to test set:
def get_KL_to_test(
    df,
    train_img,
    train_labels,
    test_img,
    test_labels,
    dist_func,
    cross_dist_func,
    dist_name,
    q_list,
    hd_list,
    test_dist = None,
):
    df = df.copy()
    num_classes = np.max(list(train_labels))+1
    KL_lists = {
        'Alpha':{hd: {q:[] for q in q_list} for hd in hd_list},
        'Gamma':{hd: {q:[] for q in q_list} for hd in hd_list}
    }
    n_test = test_img.shape[0]
    if test_dist is None:
        test_dist = dist_func(test_img)
    test_counts = torch.zeros((len(test_labels), num_classes))
    ixs_list = [[i for i in range(len(test_labels)) if test_labels[i]==c] for c in range(num_classes)]
    for i in range(num_classes):
        for j in ixs_list[i]:
            test_counts[j][i] = 1/len(test_labels)
    for ixs in tqdm(df['subset indices']):
        imgs = train_img[ixs]
        labels = train_labels[ixs]
        ixs_list = [[i for i in range(len(labels)) if labels[i]==c] for c in range(num_classes)]
        subset_counts = torch.zeros((len(labels)+len(test_labels),num_classes))
        this_test_counts = torch.cat([torch.zeros((len(labels), num_classes)), test_counts],0)
        for i in range(num_classes):
            for j in ixs_list[i]:
                subset_counts[j][i] = 1/len(ixs)
        n_subset = imgs.shape[0]
        dist = torch.eye(n_subset + n_test)
        dist[n_subset:,n_subset:] = torch.Tensor(test_dist)
        dist[:n_subset,:n_subset] = torch.Tensor(dist_func(imgs))
        dist[:n_subset, n_subset:] = cross_dist_func(imgs, test_img)
        dist[n_subset:, :n_subset] = dist[:n_subset, n_subset:].t()
        for hd in hd_list:
            sim = torch.Tensor(2**(-dist/hd))
            Zp_ratio_alpha = (sim @ this_test_counts)/(sim @ subset_counts)
            Zp_ratio_gamma = (sim @ (this_test_counts.sum(-1)))/(sim @ (subset_counts.sum(-1)))
            for q in q_list:
                if q==1:
                    alpha_KL = this_test_counts * torch.log(Zp_ratio_alpha)
                    alpha_KL = alpha_KL.sum(1).sum(0)
                    #alpha_KL = torch.exp(alpha_KL)
                    gamma_KL = this_test_counts * torch.log(Zp_ratio_alpha)
                    gamma_KL = gamma_KL.sum()
                    #gamma_KL = torch.exp(gamma_KL)
                else:
                    alpha_KL = this_test_counts * (Zp_ratio_alpha**(q-1))
                    alpha_KL = torch.logsumexp(torch.log(alpha_KL.flatten()),0)/(q-1)
                    gamma_KL = this_test_counts.sum(-1) * (Zp_ratio_gamma**(q-1))
                    gamma_KL = torch.logsumexp(torch.log(gamma_KL.flatten()),0)/(q-1)
                KL_lists['Alpha'][hd][q].append(float(alpha_KL))
                KL_lists['Gamma'][hd][q].append(float(gamma_KL))
    for hd in hd_list:
        for q in q_list:
            df[f'{dist_name}_KL_Alpha_hd={hd}_q={q}'] = KL_lists['Alpha'][hd][q]
            df[f'{dist_name}_KL_Gamma_hd={hd}_q={q}'] = KL_lists['Gamma'][hd][q]
            df = df.copy()
    return df
