# AsiDasOpenCv
ASI DAS promotional app [C++] [OpenCV]

OpenCV based face recognition demo app; adds funny overlays & animations.
Created for PWR DAS to promote ASI students' organization.

####Lunch arguments:

default config
```
--cascade="bin/haarcascade_frontalface_alt.xml" --nested-cascade="bin/haarcascade_eye_tree_eyeglasses.xml" --scale=1.0 --cam=1 --cycle-time=5
```

* -cascade - face NN nodes weights
* -nested-cascade - nested eyes NN nodes weights
* -scale - camera capture scale; affects detection results & processing time
* -cam - choose camera source (only if more cams installed on system)
* -cycle-time - overlay cycle mode interval in seconds


####Usage:

* q - quit
* d - debug mode
+ 1-4 - choose overlay
+ 0 - remove overlay
+ s - cycle through overlays



