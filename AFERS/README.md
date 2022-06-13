# AFERS - Autonomous Face and Emotion Recognition System

This project wants to create a system usable on Structures for Elderly Care in order to track the assisted emotions in a span of time.

<h2> The idea </h2>

The idea is to use the script ot create some system for the recognition and the analysis of the emotions of elderly users, both in a private environment and in Structures for Elderly Care.

<h2> P.E.R. - Proactive Emotion Recognition </h2>

After a person is recognised, their emotion is detected. If a neutral or a sad emotion is detected, the system will try to modify the elder's emotion using visual (and/or audio in the future) inputs to try to modify their emotions. If an happier emotion is detected, it is stored in a database.

<h3> Input choices</h3>

<h4>The idea for the inputs is to choose from a pool of medias at first randomly. When the elder has just finished their registration the media shown to modify their emotions is chosen randomly. After we run a few times our code and understand what the elder like, the choice will not be random.</h4>

<h3> How to choose what kind of media will be shown <h3>

<h4>The idea is to use the concept used in Machine Learning to calculate the choice of certain element in your model. We use the idea of [Laplacian Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) in order to associate proto probabilites to choose a certain kind of image. At the beginning of our learning process, it will be uniform. As we learn that a certain kind of media stimulate positive outcome on the user, the probability of choosing that will increment.</h4>

<h4>The set of media shown will not be static. The idea is to keep it somehow dynamic whilst avoiding to recalculate the probability to pick a certain media every time a new picture is added. A way to fix this is by extracting certain features or characheristic of the medias and put them as tags associated with it. By giving for each elder the probability that a certain tag will change positively their humor, we can choose from a pool of dynamic medias</h4>


<h2>Image, video and audio databases</h2>

<h4>The images and videos are fetched through API requests from [enter site](enter_site), which provides royalty-free pictures and movies. The license these medias are shared with is [CC-0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)</h4>