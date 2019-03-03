## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted]: ./output_images/distorted_image.jpg "Distorted"
[undistorted]: ./output_images/undistorted_image.jpg "Undistorted"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup

### Project Structure

The project consists of two parts, the scripts contained in the directory `pipeline_components` and the main notebook `advanced_lane_lines.ipynb`. The main notebook imports the `pipeline_components` and uses them to build the lane detection pipeline. The pipeline is built using the sklearn pipelines and by passing data along with a state which captures information in every iteration step.

Here is a description of each of the `pipeline_components`:
- `distortion_corrector`: Corrects image distortion using the camera calibration tools of openCV. The `fit` method of the transformer accepts a list of chessboard images used to fit the calibration parameters. During the transform step, the image distortion is undone.
- `image_thresholder`: Accepts a function which describes a way of thresholding images and then applies it during the transform step. It also contains common building blocks for the thresholding function as static methods.
- `perspective_transformer`: Converts the image to an bird eye view by mapping the edges of a trapezoid to the edges of a rectangle. The component takes the flag `inverse` which defines the inverse transform (used mainly for visual testing).
- `lane_approximator`: This is the largest transformer as it isolates the lanes from the preprocessed image. It uses two types of approximation, one via the `LanePixelsFinder` which isolates the lanes using consecutive windows and the `LanePixelsFinderFromPolynomial` which assumes that a polynomial has already been fitted and the lane is isolated using a margin around the curve defined by the polynomial.
- `lane_information_extractor`: This component enriches the lane information with its curvature and the distance of the car from the middle of the lanes.
- `lane_annotator`: This final component uses the information extracted from the image to annotate it with the area around the two lanes and the curvature and position information.

In addition to image data flowing through the pipeline, a global state is passed. The form of this `stateful_data` is the following:
```python
{
    'data': '<image or image information data>', 
    'cached_image': '<image cached through the pipeline components and is returned after annotation>', 
    'steps': [
        SingleStepState(
            step_number = '<counts the frames passed through the pipeline>'
            left_fit = '<fit of the left lane polynomial during the step>'
            right_fit = '<fit of the right lane polynomial during the step>'
            errors = ['error caught during the step', ...]),
        SingleStepState(...),
        ...
    ]}
```

The functionalities of the state are the following:
- `data`: Pass the output of a pipeline step as the input of the next step.
- `cached_image`: Cache the image which will be printed on the final step. This is mainly used to enrich the image during the intermediate steps with their outputs in order to use this for debugging.
- `steps`: Save information relevant to each individual step of running the pipeline (passing a single frame). This information is used to smoothen the fitting of the lanes and gather errors happening during each step.

### Pipeline (single images)

### 1. Camera Calibration

The camera calibration happens in the `DistortionCorrector` transformer. One can configure the shape of the chess corners used to fit the openCV calibration transform, in our case it is set by default to `(9, 6)`. In the main notebook, the corrector is trained using the calibration images.

This is an example of undoing the distortion:

![alt text][distorted]
![alt_text][undistorted]

The code of the component is in `pipeline_components/distortion_corrector.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
