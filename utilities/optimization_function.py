import numpy as np
import cv2
from skimage.transform import AffineTransform
from scipy.spatial.distance import cdist


def calculate_histogram(image, mask, center, n_bins, size):
    """
    Calculates the color histograms (red, green, blue channels) for a specific region of the image,
    defined by a mask and centered on the given coordinates.

    This function extracts a region of interest (ROI) from the image based on the mask and the
    center coordinates. It then computes separate histograms for the red, green, and blue channels
    of the ROI, and concatenates the results.

    Args:
        image (np.ndarray): The input image as a 3D numpy array of shape (height, width, 3),
            where the third dimension represents the color channels (RGB).
        mask (np.ndarray): A 2D numpy array of shape (height, width) with binary values (0 or 1),
            indicating the region of interest to be analyzed.
        center (tuple): A tuple (x, y) representing the center coordinates of the region of interest.
        n_bins (int): The number of bins to use for the histograms (per channel).
        size (int): The size of the square region around the center to consider when extracting
            the sub-image.

    Returns:
        np.ndarray: A 1D numpy array containing the concatenated histograms of the red, green,
            and blue channels, with a total length of `n_bins * 3`.
    """
    # set up functions to calculate colour histograms
    # function that computes the histograms for red green and blue channel of a certain region of the given
    # image. The region position is given by the mask
    x, y = center
    Mx, My = mask.shape
    x1, x2 = np.maximum(x - size // 2, 0), np.minimum(x + size // 2, Mx)
    y1, y2 = np.maximum(y - size // 2, 0), np.minimum(y + size // 2, My)
    mask = mask[x1:x2, y1:y2]
    sub_image = image[x1:x2, y1:y2]
    sub_image = sub_image.reshape([-1, 3])[mask.reshape([-1]) == 1]

    r_hist = np.histogram(sub_image[:, 0], n_bins, range=[0, 256], density=True)[0]
    g_hist = np.histogram(sub_image[:, 1], n_bins, range=[0, 256], density=True)[0]
    b_hist = np.histogram(sub_image[:, 2], n_bins, range=[0, 256], density=True)[0]

    out = np.concatenate([r_hist, g_hist, b_hist])
    return out


def par_to_H(theta, tx, ty):
    """
    Converts a set of transformation parameters into a homography matrix.

    This function creates an affine transformation matrix using the input parameters
    for rotation and translation.

    Args:
        theta (float): Rotation angle in radians.
        tx (float): Translation in the x direction.
        ty (float): Translation in the y direction.

    Returns:
        np.ndarray: A 3x3 homography matrix representing the affine transformation.
    """
    H = AffineTransform(
        scale=1, rotation=theta, shear=None, translation=[tx, ty])
    return H.params


def M_to_quadrant_dict(M, quadrants, anchor):
    """
      Generates transformation matrices for each quadrant, excluding the anchor quadrant.

      The function applies the parameters from the list `M` to each quadrant except the anchor
      and returns a dictionary where the keys are the quadrant identifiers and the values
      are the homography matrices.

      Args:
          M (list or np.ndarray): A list of parameters [theta, tx, ty] for each quadrant
              (except the anchor).
          quadrants (list): List of quadrants identifiers (e.g., [0, 1, 2, 3]).
          anchor (int or str): The quadrant to be excluded from the transformation.

      Returns:
          dict: A dictionary where the keys are the quadrant identifiers and the values
          are the corresponding 3x3 homography matrices.
      """
    H_dict = {}
    Q = [q for q in quadrants if q != anchor]
    for i, q in enumerate(Q):
        H_dict[q] = par_to_H(*[M[i] for i in range(i * 3, i * 3 + 3)])
    return H_dict


def warp(coords, H):
    """
       Applies a homography transformation to a set of coordinates.

       This function takes a set of 2D coordinates and a homography matrix,
       and applies the transformation to generate the warped coordinates.

       Args:
           coords (np.ndarray): A 2D array of shape (N, 2), where N is the number of points.
           H (np.ndarray): A 3x3 homography matrix for the transformation.

       Returns:
           np.ndarray: A 2D array of shape (N, 2) with the transformed coordinates.
       """
    # function that does the transformation
    out = cv2.perspectiveTransform(
        np.float32(coords[:, np.newaxis, :]), H)[:, 0, :]
    return out


def loss_fn(M, quadrants, anchor, data_dict, histogram_dists, max_size, alpha=0.1, d=32):
    """
        Computes the loss function for image stitching based on homography transformations.

        This function calculates a combined loss for misalignment and histogram distance between
        quadrants of an image. The function performs transformations using homographies on each
        quadrant and computes the loss between corresponding fragment edges.

        Args:
            M (list or np.ndarray): A list of parameters for homography matrices. Each set of
                three parameters is converted into a homography matrix for each quadrant.
            quadrants (list): List of quadrant identifiers.
            anchor (int or str): The quadrant to keep fixed as a reference (not transformed).
            data_dict (list of dicts): A list where each element contains data for a quadrant,
                including position lines, points, and the quadrant identifier.
            histogram_dists (np.ndarray): A matrix of histogram distances between quadrants,
                where histogram_dists[i, j]["pos_ant"] contains the distance between
                the 'pos_points' of quadrant i and the 'ant_points' of quadrant j.
            max_size (float): The normalization factor for the size of the image/region,
                used to normalize the misalignment loss.
            alpha (float, optional): A weighting factor to balance the misalignment and histogram
                losses. Default is 0.1.
            d (int, optional): A distance threshold for nearby points in the stitching process.
                Default is 32.

        Returns:
            float: The total loss, combining both the misalignment loss and the histogram loss.
    """
    # alpha is a scaling factor for the misalignment loss
    hist_loss = 0.
    mis_loss = 0.

    # H_dict contains as a dictionary the transformation matrices for each quadrant excluding the anchor
    H_dict = M_to_quadrant_dict(M, quadrants, anchor)

    for idx in range(len(data_dict)):
        # for each quadrant retrieves tha parameters
        # for each pair of fragment (quadrant) with indces i and j
        i, j = idx, (idx + 1) % len(data_dict)

        data1 = data_dict[i]
        data2 = data_dict[j]
        q1 = data1['quadrant']
        q2 = data2['quadrant']

        axis1 = data1['pos_line']
        axis2 = data2['ant_line']
        points1 = data1['pos_points']
        points2 = data2['ant_points']
        hist_dist = histogram_dists[i, j]["pos_ant"]

        # if index q1 quadrant is not the anchor retrieve the transformed edges of the fragment and the
        # transformed points coordinates
        if q1 in H_dict and q1 != anchor:
            axis1 = warp(axis1, H_dict[q1])
            points1 = warp(points1, H_dict[q1])
        if q2 in H_dict and q2 != anchor:
            axis2 = warp(axis2, H_dict[q2])
            points2 = warp(points2, H_dict[q2])

        # term of the final loss function related to the misalignment
        mis_loss += np.mean((axis1 / max_size - axis2 / max_size) ** 2)

        # samples some indices (1/4 of the total number of points) on the two fragment edges
        # (note that points1 are the 'pos_points' on fragment1 and points2 are the 'ant_points' on fragment2,
        # so they are the points that should match during the stitching)
        ss1 = np.random.choice(len(points1),
                               size=len(points1) // 4,
                               replace=False)
        ss2 = np.random.choice(len(points2),
                               size=len(points2) // 4,
                               replace=False)
        point_distance = cdist(points1[ss1], points2[ss2])
        nx, ny = np.where(point_distance < d)
        nx, ny = ss1[nx], ss2[ny]
        hnb = hist_dist[nx, ny]
        if hnb.size > 0:
            h = np.mean(hnb)
        else:
            h = 0.
        # penalty if points have no nearby pixels
        max_dist = hist_dist.max()
        no_nearby_x = point_distance.shape[0] - np.unique(nx).size
        no_nearby_x = no_nearby_x / point_distance.shape[0]
        no_nearby_y = point_distance.shape[1] - np.unique(ny).size
        no_nearby_y = no_nearby_y / point_distance.shape[1]
        hist_loss += h + (no_nearby_x + no_nearby_y) * max_dist

    # print("mis_loss: ", mis_loss)
    # print("hist_loss: ", hist_loss)
    loss = (mis_loss) * alpha + (1 - alpha) * hist_loss
    return loss


def reconstruct_mosaic(H_dict, anchor, data_dict, output_size):

    # Al momento non utilizzata, ma se ne può parlare
    # Create an empty canvas large enough to hold the stitched image
    canvas_height = output_size[0] * 2
    canvas_width = output_size[1] * 2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Loop over all the quadrants
    for idx, data in enumerate(data_dict):
        img = data['image']
        quadrant = data['quadrant']

        # Debug: Print current homography matrix
        print(f"Quadrant: {quadrant}")
        if quadrant in H_dict:
            print(f"Homography matrix for quadrant {quadrant}:")
            print(H_dict[quadrant])

            # Apply homography if it's not the anchor
            H = np.array(H_dict[quadrant], dtype=np.float32)  # Ensure it's np.float32
            try:
                warped_img = cv2.warpPerspective(img, H, (canvas_width, canvas_height))
            except cv2.error as e:
                print(f"Error warping quadrant {quadrant}: {e}")
                continue
        else:
            # Anchor image doesn't get transformed
            warped_img = img

        # Calculate the new position of the warped image based on the homography matrix
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype="float32").reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H) if quadrant in H_dict else pts

        # Get the bounding box of the transformed points
        x_min, y_min = np.int32(dst.min(axis=0).ravel())
        x_max, y_max = np.int32(dst.max(axis=0).ravel())

        # Adjust the bounding box to fit the canvas
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(canvas_width, x_max)
        y_max = min(canvas_height, y_max)

        # Place the warped image onto the canvas
        canvas[y_min:y_max, x_min:x_max] = warped_img[:(y_max - y_min), :(x_max - x_min)]

    return canvas


def cb_work_in_progress(xk, convergence):
    # Compute the loss as before
    loss_value = loss_fn(xk, quadrant_list, anchor, data_dict, histogram_dists, output_size[0] / 100, 0.1,
                         square_size // 2)
    progress.append(loss_value)

    # Generate the homography dictionary from the current parameters
    # H_dict = M_to_quadrant_dict(xk, quadrant_list, anchor)

    # Reconstruct the mosaic with current transformations
    # mosaic = reconstruct_mosaic(H_dict, anchor, data_dict)

    # Save the mosaic image for this iteration
    # iteration_num = len(progress)
    # mosaic_filename = os.path.join(root_folder, 'debug', f'mosaic_iter_{iteration_num}.png')
    # cv2.imwrite(mosaic_filename, mosaic)

    # Optionally display the mosaic image (can slow down the optimization if the display is too frequent)
    # plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    # plt.title(f'Iteration {iteration_num} - Loss: {loss_value}')
    # plt.show()