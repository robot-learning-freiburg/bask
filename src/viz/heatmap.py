def single_image_pair_keypoint_qualitative_analysis(dcn, dataset, keypoint_data_a,
                                                    keypoint_data_b,
                                                    heatmap_kernel_variance=0.25,
                                                    blend_weight_original_image=0.3,
                                                    plot_title="Keypoints"):
    """
    Wrapper for qualitative analysis of a pair of images using keypoint annotations
    :param dcn:
    :type dcn:
    :param dataset:
    :type dataset:
    :param keypoint_data_a: pandas Series
    :type keypoint_data_a:
    :param keypoint_data_b:
    :type keypoint_data_b:
    :return:
    :rtype:
    """
    DCE = DenseCorrespondenceEvaluation

    image_height, image_width = dcn.image_shape

    scene_name_a = keypoint_data_a['scene_name']
    img_a_idx = keypoint_data_a['image_idx']
    uv_a = (keypoint_data_a['u'], keypoint_data_a['v'])
    uv_a = DCE.clip_pixel_to_image_size_and_round(
        uv_a, image_width, image_height)

    scene_name_b = keypoint_data_b['scene_name']
    img_b_idx = keypoint_data_b['image_idx']
    uv_b = (keypoint_data_b['u'], keypoint_data_b['v'])
    uv_b = DCE.clip_pixel_to_image_size_and_round(
        uv_b, image_width, image_height)

    rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose(scene_name_a, img_a_idx)

    rgb_b, _, mask_b, _ = dataset.get_rgbd_mask_pose(scene_name_b, img_b_idx)

    mask_a = np.asarray(mask_a)
    mask_b = np.asarray(mask_b)

    # compute dense descriptors
    rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
    rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

    # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
    res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
    res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

    best_match_uv, best_match_diff, norm_diffs = \
        DenseCorrespondenceNetwork.find_best_match(
            uv_a, res_a, res_b, debug=False)

    # visualize image and then heatmap
    diam = 0.03
    dist = 0.01
    kp1 = []
    kp2 = []
    kp1.append(cv2.KeyPoint(uv_a[0], uv_a[1], diam))
    kp2.append(cv2.KeyPoint(best_match_uv[0], best_match_uv[1], diam))

    matches = []  # list of cv2.DMatch
    matches.append(cv2.DMatch(0, 0, dist))

    gray_a_numpy = cv2.cvtColor(np.asarray(rgb_a), cv2.COLOR_BGR2GRAY)
    gray_b_numpy = cv2.cvtColor(np.asarray(rgb_b), cv2.COLOR_BGR2GRAY)
    img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2,
                           matches, flags=2, outImg=gray_b_numpy, matchColor=(255, 0, 0))

    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    axes[0].imshow(img3)
    axes[0].set_title(plot_title)

    # visualize the heatmap
    heatmap_color = vis_utils.compute_gaussian_kernel_heatmap_from_norm_diffs(
        norm_diffs, heatmap_kernel_variance)

    # convert heatmap to RGB (it's in BGR now)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    alpha = blend_weight_original_image
    beta = 1-alpha
    blended = cv2.addWeighted(np.asarray(
        rgb_b), alpha, heatmap_color_rgb, beta, 0)

    axes[1].imshow(blended)
    return rgb_a, rgb_b
