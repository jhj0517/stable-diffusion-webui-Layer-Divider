CSS = """
#md_project a {
  color: black;
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}
"""


PROJECT_NAME = """
# [Layer-Divider-WebUI](https://github.com/jhj0517/Layer-Divider-WebUI)
"""

PARAMS_EXPLANATION = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }
    th {
      background-color: #f2f2f2;
    }
  </style>
</head>
<body>

<details>
  <summary>Explanation of Each Parameter</summary>
  <table>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>points_per_side</td>
        <td>The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.</td>
      </tr>
      <tr>
        <td>pred_iou_thresh</td>
        <td>A filtering threshold in [0,1], using the model's predicted mask quality.</td>
      </tr>
      <tr>
        <td>stability_score_thresh</td>
        <td>A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.</td>
      </tr>
      <tr>
        <td>crops_n_layers</td>
        <td>If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.</td>
      </tr>
      <tr>
        <td>crop_n_points_downscale_factor</td>
        <td>The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.</td>
      </tr>
      <tr>
        <td>min_mask_region_area</td>
        <td>If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.</td>
      </tr>
    </tbody>
  </table>
</details>

</body>
</html>
"""