/* @Author: YueLin */

/* Note: I don't understand the principle of ByteTrack,
   this header file just repackes the 'byte_track' folder.
   All codes in the 'mot/byte_tracker' folder are not implemented by me. */

#include <cmath>
#include <vector>

#include "mot/object.hpp"

#include "mot/byte_track/ByteTracker.h"  // This code is not implemented by me 

namespace mot
{
    typedef byte_track::STrack Box;
    typedef std::vector<Box> Boxes;
    typedef std::vector<int> Array;
    typedef byte_track::BYTETracker ByteTracker;

    const int FRAMES = 30;

    inline bool eq(float f1, float f2)
    {
        return std::fabs(f1 - f2) < 1e-6;
    }

    class ByteTrack
    {
        public:
            int frame;

        private:
            float thresholds[2];
            ByteTracker tracker;
        
        public:
            ByteTrack(float, float);
        
        public:
            void reset();
            Boxes track(Objects&);
            Array match(Objects&, Boxes&);
    };

    ByteTrack::ByteTrack(float thresholds1, float thresholds2)
    {
        thresholds[0] = thresholds1;
        thresholds[1] = thresholds2;
    }

    void ByteTrack::reset()
    {
        frame = 0;
        tracker = ByteTracker();
        tracker.match_thresh = thresholds[1];
        tracker.track_thresh = tracker.high_thresh = thresholds[0];
    }

    Boxes ByteTrack::track(Objects& objects)
    {
        frame++;
        return tracker.update(objects);
    }

    Array ByteTrack::match(Objects& objects, Boxes& boxes)
    {
        const int n = boxes.size();
        const int numbers = objects.size();
        Array labels(n, -1);
        for(int b = 0; b < n; b++)
        {
            Box& box = boxes[b];
            float distance = 99,
            x = box.tlwh[0] + box.tlwh[2] / 2,
            y = box.tlwh[1] + box.tlwh[3] / 2;
            for(int obj = 0; obj < numbers; obj++)
            {
                Object& object = objects[obj]; float
                x0 = object.rect.x + object.rect.width / 2,
                y0 = object.rect.y + object.rect.height / 2;
                if(std::fabs(x + y - x0 - y0) < distance)
                    labels[b] = object.label;
            }
        }
        return labels;
    }
}
