module centroidtracker;

import std.math : sqrt;
import std.algorithm.sorting : sort;
import std.typecons : Tuple, tuple;
import std.container.array : Array;
import std.container.dlist, std.range : walkLength;
import std.array : Appender;
import std.algorithm.setops : setDifference;

alias CenterCoord = Tuple!(int, int);
alias Box = int[4];

struct CentroidTracker {
    @disable this();
public:
    this(int maxDisappeared){
        this.nextObjectID = 0;
        this.maxDisappeared = maxDisappeared;
    }

    void register_Object(int cX, int cY, Box b){
        int object_ID = this.nextObjectID;
        this.objects ~= tuple(object_ID, tuple(cX, cY), b);
        this.disappeared[object_ID] = 0;
        this.nextObjectID += 1;
    }

    void update(ref Array!Box boxes){
        if (!boxes.length) {
            
            auto ks = Appender!(int[])();
            
            foreach (k, v; this.disappeared) {
                this.disappeared[k]++;
                if (v > this.maxDisappeared){
                    for (auto rn = objects[]; !rn.empty;)
                        if (rn.front[0] == k)
                            objects.popFirstOf(rn);
                        else
                            rn.popFront();
                    path_keeper[k].clear;
                    path_keeper.remove(k);
                    ks.put(k);
                }
            }

            foreach (k; ks)
                this.disappeared.remove(k);
            
            return;// this.objects;
        }

        // initialize an array of input centroids for the current frame
        Array!(Tuple!(CenterCoord, Box*)) inputCentroids; inputCentroids.length = boxes.length;
        scope(exit) inputCentroids.clear();

        foreach (i, ref b ; boxes.data) {
            int cX = cast(int)((b[0] + b[2]) / 2.0);
            int cY = cast(int)((b[1] + b[3]) / 2.0);
            inputCentroids[i] = tuple(tuple(cX, cY), &b);
        }

        //if we are currently not tracking any objects take the input centroids and register each of them
        if (this.objects[].empty) {
            foreach (ref coord_boxptr; inputCentroids) {
                this.register_Object(coord_boxptr[0][0], coord_boxptr[0][1], *coord_boxptr[1]);
            }
        }

            // otherwise, there are currently tracking objects so we need to try to match the
            // input centroids to existing object centroids
        else {
            const _len = walkLength(objects[]);
            Array!int objectIDs; objectIDs.length = _len;
            Array!CenterCoord objectCentroids; objectCentroids.length = _len;
            scope(exit){
                objectIDs.clear;
                objectCentroids.clear;
            } 
            size_t oi;
            foreach (ref ob; objects[]){
                objectIDs[oi] = ob[0];
                objectCentroids[oi] = ob[1];
                oi++;
            }

    //        Calculate Distances
            Array!(Array!float) Distances; Distances.length = objectCentroids.length;
            scope(exit){
                foreach (a; Distances.data)
                {
                    a.clear();
                }
                Distances.clear;
            }
            foreach (size_t i; 0..objectCentroids.length) {
                Array!float temp_D; temp_D.length = inputCentroids.length;
                foreach (size_t j; 0..inputCentroids.length) {
                    const dist = calcDistance(objectCentroids[i][0], objectCentroids[i][1], inputCentroids[j][0][0],
                                            inputCentroids[j][0][1]);

                    temp_D[j] = cast(float)dist;
                }
                Distances[i] = temp_D;
            }

            // load rows and cols
            Array!size_t cols; cols.length = Distances.length;
            Array!size_t rows;

            scope(exit){
                rows.clear;
                cols.clear;
            }

            //find indices for cols
            foreach (c, v; Distances.data) {
                const temp = findMin(v);
                cols[c] = temp;
            }

            //rows calculation
            //sort each mat row for rows calculation
            Array!(Array!float) D_copy; D_copy.length = Distances.length;
            foreach (i, v; Distances.data) {
                v.data.sort();
                D_copy[i] = v;
            }

            // use cols calc to find rows
            // slice first elem of each column
            
            Array!(Tuple!(float, int)) temp_rows; temp_rows.length = D_copy.length;
            scope(exit) D_copy.clear;
            
            foreach (i, el; D_copy.data) {
                temp_rows[i] = tuple(el[0], cast(int)i);
            }
            //print sorted indices of temp_rows
            rows.length = temp_rows.length;
            foreach (i, ref f_i ; temp_rows.data) {
                rows[i] = f_i[1];
            }

            temp_rows.clear;

            bool[size_t] usedRows;
            bool[size_t] usedCols;

            //loop over the combination of the (rows, columns) index tuples
            for (size_t i = 0; i < rows.length; i++) {
                //if we have already examined either the row or column value before, ignore it
                if (rows[i] in usedRows || cols[i] in usedCols) { continue; }
                //otherwise, grab the object ID for the current row, set its new centroid,
                // and reset the disappeared counter
                int objectID = objectIDs[rows[i]];

                foreach (ref id_coord_box ; objects[]){
                    if (id_coord_box[0] == objectID) {
                        id_coord_box[1][0] = inputCentroids[cols[i]][0][0];
                        id_coord_box[1][1] = inputCentroids[cols[i]][0][1];
                        id_coord_box[2] = *inputCentroids[cols[i]][1]; // update rectangle for new position
                    }
                }
                this.disappeared[objectID] = 0;

                usedRows[rows[i]] = true;
                usedCols[cols[i]] = true;
            }

            // compute indexes we have NOT examined yet
            import std.range : iota;

            auto objRows = iota(0, cast(int)objectCentroids.length);
            auto inpCols = iota(0, cast(int)inputCentroids.length);

            import std.array;
            Array!int unusedRows = Array!int(setDifference(objRows, usedRows.byKey.array.sort));
            Array!int unusedCols = Array!int(setDifference(inpCols, usedCols.byKey.array.sort));
            scope(exit){
                unusedRows.clear;
                unusedCols.clear;
            }
            //If objCentroids > InpCentroids, we need to check and see if some of these objects have potentially disappeared
            if (objectCentroids.length >= inputCentroids.length) {
                // loop over unused row indexes

                foreach (row; unusedRows) {
                    int objectID = objectIDs[row];
                    this.disappeared[objectID] += 1;

                    if (this.disappeared[objectID] > this.maxDisappeared) {

                        for (auto rn = objects[]; !rn.empty;)
                            if (rn.front[0] == objectID)
                                objects.popFirstOf(rn);
                            else
                                rn.popFront();

                        path_keeper[objectID].clear;
                        path_keeper.remove(objectID);
                        disappeared.remove(objectID);
                    }
                }
            } else {
                foreach (col; unusedCols) {
                    this.register_Object(inputCentroids[col][0][0], inputCentroids[col][0][1], *inputCentroids[col][1]);
                }
            }
        }
        //loading path tracking points
        if (!objects[].empty) {
            foreach (ref id_coord_box ; objects[]){
                auto id = id_coord_box[0];
                auto coord = id_coord_box[1];
                
                if(id in path_keeper)
                {
                    if (path_keeper[id].length > 30) {
                        path_keeper[id].clear;
                    }

                    path_keeper[id] ~= tuple(coord[0], coord[1]);
                }else
                    path_keeper[id] = Array!(CenterCoord)([tuple(coord[0], coord[1])]);
                
            }
        }
    }

    ref DList!(Tuple!(int, CenterCoord, Box)) getObjects() return {
        return objects;
    }

    ref Array!(CenterCoord)[int] getPathKeeper() return {
        return path_keeper;
    }
    
private:
    // ID, centroids, Box
    private DList!(Tuple!(int, CenterCoord, Box)) objects;

    //make buffer for path tracking
    Array!(CenterCoord)[int] path_keeper;

    int maxDisappeared;

    int nextObjectID;

    static double calcDistance(double x1, double y1, double x2, double y2){
        double x = x1 - x2;
        double y = y1 - y2;
        double dist = sqrt((x * x) + (y * y));       //calculating Euclidean distance

        return dist;
    }

    // <ID, count>
    int[int] disappeared;
}

private size_t findMin(A)(const A v, size_t pos = 0) {
    if (v.length <= pos) return (v.length);
    size_t min = pos;
    for (size_t i = pos + 1; i < v.length; i++) {
        if (v[i] < v[min]) min = i;
    }
    return min;
}