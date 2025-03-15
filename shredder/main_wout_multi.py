import os
import sys
import pathlib
import argparse
import tqdm
import numpy as np
import pyvips
import matplotlib.pyplot as plt
import cv2
import copy
import json

from shapely.geometry import LineString, Point

from line_utils import interpolate_contour, apply_im_tform_to_coords

def collect_arguments():
    """
    Parse arguments formally.
    """
    parser = argparse.ArgumentParser(
        description="convert svs to tif (without multiresolutionimageinterface)"
    )
    parser.add_argument(
        "--datadir", required=True, type=pathlib.Path, help="Path with the tiffs to shred"
    )
    parser.add_argument(
        "--maskdir", required=True, type=pathlib.Path, help="Path with the tissuemasks"
    )
    parser.add_argument(
        "--savedir", required=True, type=pathlib.Path, help="Path to save the shreds"
    )
    parser.add_argument(
        "--rotation", required=False, type=int, default=5, help="Random rotation of the mount before shredding"
    )
    parser.add_argument(
        "--fragments", required=False, type=int, default=4, help="Number of fragments to shred to"
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.datadir)
    mask_dir = pathlib.Path(args.maskdir)
    save_dir = pathlib.Path(args.savedir)
    rotation = args.rotation
    n_fragments = args.fragments

    assert data_dir.exists() and data_dir.is_dir(), "provided data location doesn't exist"
    assert mask_dir.exists() and mask_dir.is_dir(), "provided mask location doesn't exist"
    assert rotation in np.arange(0, 26), "rotation must be in range [0, 25]"
    assert n_fragments in [2, 4], "number of fragments must be either 2 or 4"

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    print(
        f"\nRunning job with parameters:"
        f"\n - Data dir: {data_dir}"
        f"\n - Tissue mask dir: {mask_dir}"
        f"\n - Save dir: {save_dir}"
        f"\n - Rotation: {rotation}"
        f"\n - Number of fragments: {n_fragments}\n"
    )

    return data_dir, mask_dir, save_dir, rotation, n_fragments


class Shredder:
    def __init__(self, case, mask_dir, save_dir, rotation, n_fragments):
        self.case = case
        self.mask_path = mask_dir.joinpath(f"{self.case.stem}.tif")
        self.savedir = save_dir.joinpath(self.case.stem)
        self.rotation = rotation
        self.n_fragments = n_fragments
        self.lowres_downscale = 32  # adjust this factor as needed
        self.pad_factor = 0.3
        self.n_samples = 10
        self.noise = 20
        self.step = 50

        self.parameters = {"rotation": self.rotation, "n_fragments": self.n_fragments}

        if not self.savedir.is_dir():
            self.savedir.mkdir(parents=True)
            self.savedir.joinpath("raw_images").mkdir()
            self.savedir.joinpath("raw_masks").mkdir()

    def load_images(self):
        """
        Load the full resolution image using pyvips and create a low resolution copy.
        """
        print(f"Loading case: {self.case}")
        # Load full-res image
        self.full_image = pyvips.Image.new_from_file(str(self.case))
        # Create a low-resolution image
        lowres = self.full_image.resize(1 / self.lowres_downscale)
        # Convert lowres image to numpy array; note that pyvips returns data as a binary buffer.
        self.lowres_image = np.ndarray(
            buffer=lowres.write_to_memory(),
            dtype=np.uint8,
            shape=[lowres.height, lowres.width, lowres.bands]
        )
        # Remove paraffin for better tissue masking later
        self.lowres_image_hsv = cv2.cvtColor(self.lowres_image, cv2.COLOR_RGB2HSV)
        sat_thres = 15
        self.sat_mask = self.lowres_image_hsv[:, :, 1] < sat_thres
        self.lowres_image[self.sat_mask] = 255

    def get_mask(self):
        """
        Get the postprocessed mask of the low-res image.
        """
        self.lowres_mask = np.all(self.lowres_image != [255, 255, 255], axis=2)
        self.lowres_mask = (self.lowres_mask * 255).astype(np.uint8)

        # Flood fill to remove holes
        self.temp_pad = int(0.05 * self.lowres_mask.shape[0])
        self.lowres_mask = np.pad(
            self.lowres_mask,
            [[self.temp_pad, self.temp_pad], [self.temp_pad, self.temp_pad]],
            mode="constant",
            constant_values=0,
        )
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        self.lowres_mask = cv2.dilate(self.lowres_mask, strel)
        seedpoint = (0, 0)
        floodfill_mask = np.zeros((self.lowres_mask.shape[0] + 2, self.lowres_mask.shape[1] + 2), np.uint8)
        _, _, self.lowres_mask, _ = cv2.floodFill(self.lowres_mask, floodfill_mask, seedpoint, 255)
        self.lowres_mask = 1 - self.lowres_mask[
            self.temp_pad+1: -(self.temp_pad+1), self.temp_pad+1: -(self.temp_pad+1)
        ]
        # Keep only largest connected component
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(self.lowres_mask, connectivity=8)
        largest_cc_label = np.argmax(stats[1:, -1]) + 1
        self.lowres_mask = ((labeled_mask == largest_cc_label) * 255).astype(np.uint8)

    def process(self):
        """
        Determine image rotation and crop mask accordingly.
        """
        temp_pad = int(self.pad_factor * np.min(self.lowres_mask.shape))
        temp_mask = np.pad(
            self.lowres_mask,
            [[temp_pad, temp_pad], [temp_pad, temp_pad]],
            mode="constant", constant_values=0
        )
        cnt, _ = cv2.findContours(temp_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cnt = np.squeeze(max(cnt, key=cv2.contourArea))
        bbox = cv2.minAreaRect(cnt)
        angle = bbox[2]
        if angle > 45: angle -= 90
        angle_noise = np.random.randint(-self.rotation, self.rotation)
        self.angle = int(angle + angle_noise)
        rot_center = (0, 0)
        rot_mat = cv2.getRotationMatrix2D(rot_center, self.angle, 1)
        temp_mask = cv2.warpAffine(temp_mask, rot_mat, temp_mask.shape[::-1])
        temp_mask = (temp_mask > 128) * 255
        self.r, self.c = np.nonzero(temp_mask)
        self.lowres_mask = temp_mask[np.min(self.r):np.max(self.r), np.min(self.c):np.max(self.c)]

    def get_shred_parameters(self):
        """
        Create shredding masks and compute intersection.
        """
        self.offset = 5
        v_start = [int(0.5 * self.lowres_mask.shape[1]), -self.offset]
        v_end = [int(0.5 * self.lowres_mask.shape[1]), self.lowres_mask.shape[0] + self.offset - 1]
        self.v_line_y = np.arange(v_start[1], v_end[1] + self.step, step=self.step)
        self.v_line_x = [v_start[0]]
        while len(self.v_line_x) < len(self.v_line_y):
            self.v_line_x.append(self.v_line_x[-1] + np.random.randint(-self.noise, self.noise))
        self.parameters["step_size"] = self.step
        self.parameters["edge_curvature"] = self.noise

        h_start = [-self.offset, int(0.5 * self.lowres_mask.shape[0])]
        h_end = [self.lowres_mask.shape[1] + self.offset - 1, int(0.5 * self.lowres_mask.shape[0])]
        self.h_line_x = np.arange(h_start[0], h_end[0] + self.step, step=self.step)
        self.h_line_y = [h_start[1]]
        while len(self.h_line_y) < len(self.h_line_x):
            self.h_line_y.append(self.h_line_y[-1] + np.random.randint(-self.noise, self.noise))

        v_line = LineString([(x, y) for x, y in zip(self.v_line_x, self.v_line_y)])
        h_line = LineString([(x, y) for x, y in zip(self.h_line_x, self.h_line_y)])
        inter = v_line.intersection(h_line)
        self.intersection = [int(inter.x), int(inter.y)]

        self.h_line = np.array([self.h_line_x, self.h_line_y]).T
        self.h_line_temp = interpolate_contour(self.h_line)
        self.h_line_temp = [pt for pt in self.h_line_temp if 0 < pt[1] < self.lowres_mask.shape[0] and 0 < pt[0] < self.lowres_mask.shape[1]]
        self.h_line_interp = np.array([pt for pt in self.h_line_temp if self.lowres_mask[pt[1], pt[0]] == 255])

        self.v_line = np.array([self.v_line_x, self.v_line_y]).T
        self.v_line_temp = interpolate_contour(self.v_line)
        self.v_line_temp = [pt for pt in self.v_line_temp if 0 < pt[1] < self.lowres_mask.shape[0] and 0 < pt[0] < self.lowres_mask.shape[1]]
        self.v_line_interp = np.array([pt for pt in self.v_line_temp if self.lowres_mask[pt[1], pt[0]] == 255])

    def apply_shred(self):
        """
        Use shredding parameters to mark fragment boundaries.
        """
        self.shredded_mask = copy.copy(self.lowres_mask)
        h_mid_idx = np.argmin(np.sum((self.h_line_interp - self.intersection) ** 2, axis=1))
        v_mid_idx = np.argmin(np.sum((self.v_line_interp - self.intersection) ** 2, axis=1))
        offset = 25
        h_left_idx = np.linspace(offset, h_mid_idx, self.n_samples).astype(int)
        h_right_idx = np.linspace(h_mid_idx, len(self.h_line_interp) - 1 - offset, self.n_samples).astype(int)
        self.h_line_left = self.h_line_interp[h_left_idx]
        self.h_line_right = self.h_line_interp[h_right_idx]
        v_upper_idx = np.linspace(offset, v_mid_idx, self.n_samples).astype(int)
        v_lower_idx = np.linspace(v_mid_idx, len(self.v_line_interp) - 1 - offset, self.n_samples).astype(int)
        self.v_line_upper = self.v_line_interp[v_upper_idx]
        self.v_line_lower = self.v_line_interp[v_lower_idx]

        if self.n_fragments == 4:
            for line in [self.h_line_interp, self.v_line_interp]:
                self.shredded_mask[line[:, 1], line[:, 0]] = 0
        elif self.n_fragments == 2:
            self.shredded_mask[self.v_line_interp[:, 1], self.v_line_interp[:, 0]] = 0

        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.shredded_mask = self.shredded_mask.astype(np.uint8)
        self.shredded_mask = cv2.erode(self.shredded_mask, strel, iterations=1)
        self.shredded_mask = ((self.shredded_mask > 128) * 255).astype(np.uint8)

        seed_offset = 100
        if self.n_fragments == 2:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1]],
                [self.intersection[0] + seed_offset, self.intersection[1]]
            ])
        else:
            seed_points = np.array([
                [self.intersection[0] - seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] - seed_offset, self.intersection[1] + seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] - seed_offset],
                [self.intersection[0] + seed_offset, self.intersection[1] + seed_offset],
            ])
        self.mask_fragments = []
        num_labels, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(self.shredded_mask, connectivity=8)
        self.all_set_a = [self.h_line_left if pt[0] < self.intersection[0] else self.h_line_right for pt in seed_points]
        self.all_set_b = [self.v_line_upper if pt[1] < self.intersection[1] else self.v_line_lower for pt in seed_points]
        for seed in seed_points:
            label_value = labeled_mask[seed[1], seed[0]]
            fragment = ((labeled_mask == label_value) * 255).astype(np.uint8)
            self.mask_fragments.append(fragment)

        plt.figure()
        plt.title(f"Step: {self.step}, Curvature: {self.noise}")
        plt.imshow(labeled_mask, cmap="gray")
        for a, b in zip(self.all_set_a, self.all_set_b):
            plt.scatter(a[:, 0], a[:, 1], c="r")
            plt.scatter(b[:, 0], b[:, 1], c="r")
        plt.show()

    def get_shredded_images(self):
        """
        Process high-res image in the same order as low-res and write output.
        """
        # Use the already loaded full resolution image
        self.pyvips_image = self.full_image
        output_padding = int(self.pad_factor * min(self.pyvips_image.width, self.pyvips_image.height))
        output_width = self.pyvips_image.width + 2 * output_padding
        output_height = self.pyvips_image.height + 2 * output_padding
        self.pyvips_image = self.pyvips_image.gravity("centre", output_width, output_height)
        self.pyvips_mask = pyvips.Image.new_from_file(str(self.mask_path)).gravity("centre", output_width, output_height)

        rot_mat = cv2.getRotationMatrix2D(center=(0, 0), angle=self.angle, scale=1)
        rot_coeffs = [rot_mat[0, 0], rot_mat[0, 1], rot_mat[1, 0], rot_mat[1, 1]]
        self.pyvips_image = self.pyvips_image.affine(rot_coeffs, oarea=[0, 0, self.pyvips_image.width, self.pyvips_image.height])
        self.pyvips_mask = self.pyvips_mask.affine(rot_coeffs, oarea=[0, 0, self.pyvips_mask.width, self.pyvips_mask.height])

        rmin, rmax = int(self.lowres_downscale * np.min(self.r)), int(self.lowres_downscale * np.max(self.r))
        cmin, cmax = int(self.lowres_downscale * np.min(self.c)), int(self.lowres_downscale * np.max(self.c))
        width = cmax - cmin
        height = rmax - rmin
        self.pyvips_image = self.pyvips_image.crop(cmin, rmin, width, height)
        self.pyvips_mask = self.pyvips_mask.crop(cmin, rmin, width, height)

        for count, (fragment, set_a, set_b) in enumerate(zip(self.mask_fragments, self.all_set_a, self.all_set_b), 1):
            print(f"Shredding piece {count}...")
            h, w = fragment.shape
            bands = 1
            dformat = "uchar"
            fragment_norm = (fragment / (fragment.max() or 1)).astype(np.uint8)
            self.fragment = pyvips.Image.new_from_memory(fragment_norm.ravel(), w, h, bands, dformat)

            # Create white background and foreground masks
            fragment_white_bg = copy.copy(fragment_norm)
            k = max(1, int(h / 200))
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fragment_white_bg = cv2.erode(fragment_white_bg, strel)
            fragment_white_bg = ~(fragment_white_bg * 255)
            self.fragment_white_bg = pyvips.Image.new_from_memory(fragment_white_bg.ravel(), w, h, bands, dformat)
            fragment_white_fg = ((~fragment_white_bg) / 255).astype(np.uint8)
            self.fragment_white_fg = pyvips.Image.new_from_memory(fragment_white_fg.ravel(), w, h, bands, dformat)

            self.fragment = self.fragment.resize(self.lowres_downscale)
            self.fragment_white_bg = self.fragment_white_bg.resize(self.lowres_downscale)
            self.fragment_white_fg = self.fragment_white_fg.resize(self.lowres_downscale)
            set_a = set_a * self.lowres_downscale
            set_b = set_b * self.lowres_downscale

            self.fragment_mask = self.pyvips_mask.multiply(self.fragment)
            self.fragment_image = self.pyvips_image.multiply(self.fragment)

            r_coords, c_coords = np.nonzero(fragment)
            rmin_f, rmax_f = int(self.lowres_downscale * np.min(r_coords)), int(self.lowres_downscale * np.max(r_coords))
            cmin_f, cmax_f = int(self.lowres_downscale * np.min(c_coords)), int(self.lowres_downscale * np.max(c_coords))
            width_f = cmax_f - cmin_f
            height_f = rmax_f - rmin_f

            self.fragment_image = self.fragment_image.crop(cmin_f, rmin_f, width_f, height_f)
            self.fragment_mask = self.fragment_mask.crop(cmin_f, rmin_f, width_f, height_f)
            self.fragment_white_bg = self.fragment_white_bg.crop(cmin_f, rmin_f, width_f, height_f)
            self.fragment_white_fg = self.fragment_white_fg.crop(cmin_f, rmin_f, width_f, height_f)

            set_a = np.vstack([set_a[:, 0] - cmin_f, set_a[:, 1] - rmin_f]).T
            set_b = np.vstack([set_b[:, 0] - cmin_f, set_b[:, 1] - rmin_f]).T

            rot_k = np.random.randint(0, 4)
            self.parameters["rot_k"] = rot_k
            rot_set_a = apply_im_tform_to_coords(set_a, self.fragment_image, self.lowres_downscale, rot_k)
            rot_set_b = apply_im_tform_to_coords(set_b, self.fragment_image, self.lowres_downscale, rot_k)

            self.fragment_image = self.fragment_image.rotate(rot_k * 90)
            self.fragment_mask = self.fragment_mask.rotate(rot_k * 90)
            self.fragment_white_bg = self.fragment_white_bg.rotate(rot_k * 90)
            self.fragment_white_fg = self.fragment_white_fg.rotate(rot_k * 90)

            rot_sets = {"a": rot_set_a, "b": rot_set_b}
            np.save(self.savedir.joinpath(f"fragment{count}_coordinates"), rot_sets)

            spacing = 0.25
            xyres = 1000 / spacing
            self.fragment_image_save = self.fragment_image.copy(xres=xyres, yres=xyres)
            self.fragment_image_save = self.fragment_image_save + self.fragment_white_bg
            self.fragment_image_save = self.fragment_image_save.cast("uchar")

            print(" - saving image")
            self.fragment_image_save.write_to_file(
                str(self.savedir.joinpath("raw_images", f"fragment{count}.tif")),
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=80,
            )

            print(" - saving mask")
            self.fragment_mask_save = self.fragment_mask.copy(xres=xyres, yres=xyres)
            self.fragment_mask_save = self.fragment_mask_save.multiply(self.fragment_white_fg)
            self.fragment_mask_save.write_to_file(
                str(self.savedir.joinpath("raw_masks", f"fragment{count}.tiff")),
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=20,
            )

            with open(self.savedir.joinpath(f"fragment{count}_shred_parameters.json"), "w") as f:
                json.dump(self.parameters, f, ensure_ascii=False)

    def process_case(self):
        self.load_images()
        self.get_mask()
        self.process()
        self.get_shred_parameters()
        self.apply_shred()
        self.get_shredded_images()


def main():
    data_dir, mask_dir, save_dir, rotation, n_fragments = collect_arguments()
    cases = sorted(list(data_dir.iterdir()))
    print(f"Found {len(cases)} cases to shred")
    cases = [case for case in cases if not save_dir.joinpath(case.name.rstrip(".tif"), "fragment4_shred_parameters.json").exists()]
    print(f"Shredding {len(cases)} remaining cases...")
    for case in tqdm.tqdm(cases, total=len(cases)):
        shredder = Shredder(case, mask_dir, save_dir, rotation, n_fragments)
        shredder.process_case()


if __name__ == "__main__":
    main()