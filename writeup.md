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

[distorted]: ./output_images/distorted_image.jpg "Distorted"
[undistorted]: ./output_images/undistorted_image.jpg "Undistorted"

[original_test_image]: ./output_images/original_test_image.jpg "Original Test Image"
[thresholded_image]: ./output_images/thresholded_image.jpg "Thresholded image"

[perspective_transform_points]: ./output_images/perspective_transform_points.jpg "Perspective Transform Points"
[perspective_transform_transformed]: ./output_images/perspective_transform_transformed.jpg "Perspective Transform Transformed"
[perspective_transform_reversed]: ./output_images/perspective_transform_reversed.jpg "Perspective Transform Reversed"

[perspective_transform_of_thresholded_image]: ./output_images/perspective_transform_of_thresholded_image.jpg "Perspective Transform of Thresholded Image"

[lanes_approximates_with_windows]: ./output_images/lanes_approximates_with_windows.jpg "Lanes Approximates with Windows"
[lanes_estimated_around_previous_polynomial]: ./output_images/lanes_estimated_around_previous_polynomial.jpg "Lanes Estimated around Previous Polynomial"

[annotated_image]: ./output_images/annotated_image.jpg "Annotated Image"

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

#### 1. Camera Calibration

The camera calibration happens in the `DistortionCorrector` transformer. One can configure the shape of the chess corners used to fit the openCV calibration transform, in our case it is set by default to `(9, 6)`. In the main notebook, the corrector is trained using the calibration images.

This is an example of undoing the distortion:

![alt text][distorted]
![alt_text][undistorted]

The code of the component is in `pipeline_components/distortion_corrector.py`.

#### 2. Image Thresholding

As mentioned above the `ImageThresholder` component accepts a thresholding function and provides building blocks to implement it. Using those building blocks I built a simple strategy which works as follows:
- Detects white lanes:
    - Filters the image to the parts where the RGB values are close one to another i.e. gray regions which do not contain any other color
    - Looks for vertical lines with an x-axis Sobel filter with a 11 pixels wide kernel and smoothing applied afterwards to fill the contents of the borders
    - Defines white lanes as the points which are both non-colorful and look like vertical lines
    
- Detects yellow lanes:
    - Filters the image to the parts where the RGB values differ one from another i.e. colorful regions
    - Looks for vertical lines with an x-axis Sobel filter with a 11 pixels wide kernel and smoothing applied afterwards to fill the contents of the borders
    - Defines yellow lanes as the points which are both colorful and look like vertical lines
- Merges white and yellow lanes

Here is an example of a thresholded image:

![alt_text][original_test_image]
![alt_text][thresholded_image]

The code of the image thresholder can be found in `pipeline_components/image_thresholder.py`.

#### 3. Perspective Transform

The `PerspectiveTransformer` performs a perspective transform by mapping four points of the image which define the perspective to a parallelogram. The points picked are relative to the dimensions of the image. The lower edges of the trapezoid defining the perspective are on the bottom corners of the image which the upper edges lie around the middle of the horizontal line dividing the image into two equal pieces.

Here is an example of the transform together with the inverse which helps understand which part of the image is cut off:

![alt_text][perspective_transform_points]
![alt_text][perspective_transform_transformed]
![alt_text][perspective_transform_reversed]

This is how the output of the perspective transform looks like on a thresholded image:

![alt_text][perspective_transform_of_thresholded_image]

The code of the perspective transform is in `pipeline_components/perspective_transformer.py`.

#### 4. Lane Approximation with 2nd Degree Polynomial

The `LaneApproximator` is the most complicated component of the pipeline and its role is to approximate the two lanes with polynomials given a thresholded, bird eye view image. The component first isolates the points which potentially belong to the left and right lane and then fits two polynomials.

It can isolate the lane points using following two approximations:
- Window approximation: Here it uses the `LanePixelsFinder` which builds windows on the left and right side of the thresholded image using the distributions of points to initialize and a margin during each iteration step.
- Previously fitted polynomial approximation: Here the `LanePixelsFinderFromPolynomial` uses the graph of a previously fitted polynomial to select the point around it given a margin.

After the point of the two lines have been selected, the `numpy` function `polyfit` is being used to fit the polynomials using a least squares approximation method.

Here is a list of functions which provide additional features to make the detection more reliable:

- `restrict_x_in_image`: If a polynomial curve exceeds the limits of an image this restricts it to the image.
- `average_with_previous_steps`: Using the information from polynomials from previous steps this method sets the new fit to a weighted average of the previous and current polynomials.
- `check_data_existence`: Checks for the existence of enough points to fit a polynomial.
- `bad_quality_fits`: Compares the newly fitted polynomial with the previous ones and if its coefficients differ too much, then it discards it.

The `transform` method performs all the above and:
- Fits based on previous polynomials, if existent
- If this fails, it finds a lane using windows
- It smoothens the output by averaging with previous polynomials
- If all previous outputs and the current attempt fail to provide a good fit, then it returns no image

Here is an example of lanes approximated with consecutive windows and with a margin around an existing polynomial:

![alt_text][lanes_approximates_with_windows]
![alt_text][lanes_estimated_around_previous_polynomial]

The code of the lane approximation can be found in `pipeline_components/lane_approximator.py`

#### 5. Calculate Radius of Curvature and Distance from Lane Middle

The `LaneInformationExtractor` is performing both those actions using the material covered in the course. The radius of curvature is computed by averaging the results of the formulas computing the radius of curvature of each lane polynomial. The distance from the middle of the lanes is simply the distance between the middle of the image and the middle of the lower part of the estimate lanes. In both calculations the rule of thumb of `3.7 / 700` horizontal meter over pixel ratio and `30 / 720` vertical meter over pixel ratio is applied to scale to meters.

Look at the next paragraph for an example image where both the radius of curvature and distance from the lanes middle are computed.

The code of the information extractor can be found in `pipeline_components/lane_information_extractor.py`.

#### 6. Annotating the Area Between the Lanes

The final component of the pipeline is the `LaneAnnotator` which picks the information extracted from the image and annotates it. Here is an example:

![alt_text][annotated_image]

The code of the annotator can be found in `pipeline_components/lane_annotator`.

### Pipeline (video)

#### Final output

I run the above pipeline on all three videos. Here are links to them:

- [Project video](./output_videos/project_video.mp4)
- [Challenge video](./output_videos/challenge_video.mp4)
- [Harder challenge video](./output_videos/harder_challenge_video.mp4)

Here are links the same videos annotated with intermediate results such as the thresholde image and lane approximation outputs. They were very useful during debugging the pipeline.

- [Project video](./output_videos_debugging/project_video.mp4)
- [Challenge video](./output_videos_debugging/challenge_video.mp4)
- [Harder challenge video](./output_videos_debugging/harder_challenge_video.mp4)

---

### Discussion

#### 1. Existing issues and cases where the pipeline will fail

Some issues which already appear in the hard challenge are:
- The detection of white lanes is dependent to a point on the brightness of the pixels. This has as a result that the detection directly fails on darker backgrounds, e.g. under bridges or tree shadows. In specific it would totally fail on night environment.
- Again because the the detection uses brightness, the algorithm gets confused by large bight spots and fits the predicted lanes towards them.
- Any bright colorful object appearing in the image can be mistaken for a yellow lane. This could include cars, objects in the background near the lanes, additional traffic marks on the street and the sun shining toward the windscreen.
- In a similar way as in the issue above any bright white object will be mistaken for a white lane.
- Some times the predicted lanes can slowly move to a weird position e.g. converge on one of the edges of the frames. The smoothing performed lets to the fitted lanes remain in those positions for a while before resetting them using the window algorithm.
- The fitted lanes might be non-parallel or even cross each other. 

#### 1. Potential solutions to the pipeline issues
The most crucial part of the pipeline is the thresholding of the image to make a good filtering of points which are almost the lanes. While improving this component it felt like manually building a CNN. Namely the different blocks used mapped to corresponding CNN components. E.g. the Sober filter or color combinations were convolutions, the thresholding was application of ReLu and linear combinations of the above combination of filters. Using an end-to-end learning approach where one optimizes a metric like intersection over union for the lanes annotated on the images would be a much more precise and scalable solution. This will become evident as one tries to cover different driving environments and correct failure cases.

Some more direct and short-term improvements could be:

- To avoid the issues related with brightness one can come up with an alternative, more restrictive definition of a lane. Some examples are 
    - Set stricter restrictions on the thickness (this would filter the thin line in the middle in the challenge video)
    - Add more derivative directions on the Sobel filter to remove surroundings
- To avoid the convergence of the lanes to weird positions one can add stricter criteria on accepting polynomials approximating the lanes. Those could include:
    - Checking that the lanes are fairly parallel
    - Setting a threshold for the maximum squared error allowed during the fitting of the polynomials
    - Cluster the points using to fit lanes to multiple potential lanes and pick the two most relevant ones
- To make sure the two lines are fairly parallel:
    - The aforementioned check can be added to accept only such lanes
    - One can define a method computing the fitness of a fitted lane and then use the best fit (left or right lane) to define one lane and correct the other one. This method could also give a chance to drive on an environment where the one lane is missing for some reason.

#### 2. Issues regarding the codebase

- The manual tests via the cell outputs will not be enough if one further develops the codebase. Proper unit & integration tests would be needed to ensure consistency and faster development.
- The pipeline has become somewhat slower as more components have been added. One can refactor parts of the code and get faster by vectorizing operations, parallelizing some processes and remove unnecessary checks. Also a profiling of the code could reveal bottlenecks.
- A better type of pipeline than the generic pipeline of sklearn could be used to handle the internal state aggregated along the frames.

# Question to the reviewer

During this project I came up with some questions regarding best practices on building such pipelines. I state here the one that would provide the biggest value if answered:

- Which framework(s) would you suggest using to build a stateful preprocessing or machine learning pipeline? Could you point me to an example of open sourced project that makes a real-life use of it?

To elaborate a bit more on the question I can give you a link to this [recent post about building such pipelines](https://www.kdnuggets.com/2019/02/4-reasons-machine-learning-code-probably-bad.html) which triggered my interest. It suggest using DAGs (Directed Acyclic Graphs) of tasks instead of linear pipelines for machine learning projects. Some frameworks/libraries mentioned there are `d6tflow`, `luigi` and `airflow`. 

Additionally one of my challenges was keeping track of an internal state of the pipeline. In this case I chose to merge the state with the input data and then use it or over-write it when needed. It would have been great though if the pipeline had tools build for this case e.g. the state passed as argument of the transform method. This would make the separation of data and state much more explicit and building a pipeline from scratch quicker. Do you know if any of the above, or any other workflow building tool supports that?

