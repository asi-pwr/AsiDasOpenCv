#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

static void help()
{
    cout << "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );


const String windowName = "symulator nerda ASI";

string cascadeName;
string nestedCascadeName;

bool debugMode = false;
Mat overlay;

int main( int argc, const char** argv )
{
    overlay = imread ( "overlay.png", -1 );

    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryFlip;
    CascadeClassifier cascade, nestedCascade;
    double scale;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );

    if (parser.has("help")) {
        help();
        return 0;
    }

    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");

    if (scale < 1){
        scale = 1;
    }

    tryFlip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if ( !nestedCascade.load( nestedCascadeName ) ) {
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) ) {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera)) {
            cout << "Capture from camera #" << camera << " didn't work" << endl;
        }
    }
    else if( !inputName.empty() ) {
        image = imread( inputName, 1 );
        if( image.empty() ) {
            if(!capture.open( inputName )) {
                cout << "Could not read " << inputName << endl;
            }
        }
    }
    else {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) {
            cout << "Couldn't read ../data/lena.jpg" << endl;
        }
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        for(;;)
        {
            capture >> frame;

            if( frame.empty() ){
                break;
            }

            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, nestedCascade, scale, tryFlip );

            auto input = (char)waitKey(10);
            if( input == 27 || input == 'q' || input == 'Q' ) {
                break;
            }
            else if(input == 'd' || input == 'D' ) {
                debugMode = !debugMode;
            }
        }
    }
    else
    {
        cout << "Detecting face(s) in " << inputName << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryFlip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    auto len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() ) {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryFlip );

                        auto input = (char)waitKey(0);
                        if( input == 27 || input == 'q' || input == 'Q' ) {
                            break;
                        }
                        else if(input == 'd' || input == 'D' ) {
                            debugMode = !debugMode;
                        }

                    }
                    else {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }
    return 0;
}

void overlayImage(const cv::Mat &background, cv::Mat foreground,
                  cv::Mat &output, cv::Point2i location, double scale, double angle) {

    cv::Mat foreground2;

    if(scale != 1.0) {
        cv::resize(foreground, foreground, cv::Size((int) (200 * scale), (int) (200 * scale))); //resize overlay
    }

    if (angle != 0.0) {
        Point2f src_center(foreground.cols / 2.0F, foreground.rows / 2.0F); //rotate overlay
        Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
        warpAffine(foreground, foreground2, rot_mat, foreground.size());
    }

    background.copyTo(output);


    // start at the row indicated by location, or at row 0 if location.y is negative.
    for(int y = std::max(location.y , 0); y < background.rows; ++y)
    {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if(fY >= foreground2.rows)
            break;

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for(int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if(fX >= foreground2.cols)
                break;

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                    ((double)foreground2.data[fY * foreground2.step + fX * foreground2.channels() + 3])

                    / 255.;


            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for(int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                unsigned char foregroundPx =
                        foreground2.data[fY * foreground2.step + fX * foreground2.channels() + c];
                unsigned char backgroundPx =
                        background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] =
                        backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    Mat gray, smallImg;
    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );

    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );

        for( auto r = faces2.begin(); r != faces2.end(); ++r ) {
            faces.emplace_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }

    t = (double)getTickCount() - t;
    printf( "FacesCount = %d; DetectionTime = %g ms; Debug = %d\n", (int)faces.size(), t*1000/getTickFrequency(), debugMode);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        double aspect_ratio = (double)r.width/r.height;

        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            if(debugMode) {
                circle(img, center, radius, color, 3, 8, 0);
            }
        }
        else if (debugMode) {
            rectangle(img, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)),
                      cvPoint(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
                      color, 3, 8, 0);
        }

        if( nestedCascade.empty() ){
            continue;
        }

        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );

        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            if(debugMode) {
                circle(img, center, radius, color, 3, 8, 0);
            }
        }
    }

    overlayImage(img, overlay, img, cv::Point(0,0), 1.0, 120.0); //draw overlay

    if(!debugMode) {
        putText(img, "Symulator typowego ASIowicza", Point(70, 50),
                CV_FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 128), 2, LINE_AA); //draw text

    }

    resize(img, img, Size(img.cols*2, img.rows*2)); // scale window
    namedWindow( windowName,CV_WINDOW_AUTOSIZE);
    imshow( windowName, img );
}