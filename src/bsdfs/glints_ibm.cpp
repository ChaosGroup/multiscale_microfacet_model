// Author: Asen Atanasov

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/// A class representing a 2d vector
class Vector2 {
public:
    /// The components of the vector
    float x, y;

    /// Default constructor - does nothing
    Vector2() {}

    /// Initializes the vector with the given components.
    Vector2(float ix, float iy) {
        x = ix;
        y = iy;
    }

    /// Returns the i-th (0 or 1) components of the vector
    float &operator[](const int index) {
        return reinterpret_cast<float(&)[2]>(*this)[index];
    }

    /// Returns the i-th (0 or 1) components of the vector as const
    const float &operator[](const int index) const {
        return reinterpret_cast<const float(&)[2]>(*this)[index];
    }

    /// Adds the components of the given vector
    Vector2 &operator+=(const Vector2 &a) {
        x += a.x;
        y += a.y;
        return *this;
    }

    /// Multiplies the components of the vector by the given number
    Vector2 &operator*=(float f) {
        x *= f;
        y *= f;
        return *this;
    }

    /// Sets the components of the vector
    void set(float ix, float iy) {
        x = ix;
        y = iy;
    }
};

/// \relates Vector2
/// Adds the components of the given vectors
inline Vector2 operator+(const Vector2 &a, const Vector2 &b) {
    return Vector2(a.x + b.x, a.y + b.y);
}

/// \relates Vector2
/// Subtracts the components of the given vectors
inline Vector2 operator-(const Vector2 &a, const Vector2 &b) {
    return Vector2(a.x - b.x, a.y - b.y);
}

/// \relates Vector2
/// Multiplies the components of a vector by a number
inline Vector2 operator*(const Vector2 &a, float f) {
    return Vector2(a.x * f, a.y * f);
}

/// \relates Vector2
/// Return the dot product of two Vector2's; calculations are single-precision.
inline float dotf(const Vector2 &a, const Vector2 &b) {
    return a.x * b.x + a.y * b.y;
}

/// \relates Vector2
/// Return the cross product of two Vector2's; calculations are
/// single-precision.
inline float crossf(const Vector2 &a, const Vector2 &b) {
    return a.x * b.y - a.y * b.x;
}

/// \relates Vector2
/// Reverses the sign of all components
inline Vector2 operator-(const Vector2 &a) { return Vector2(-a.x, -a.y); }

/// \relates Vector2
/// Returns a vector with components which are the minimum respective components
/// of the two vectors
inline Vector2 Min(const Vector2 &a, const Vector2 &b) {
    return Vector2(std::min(a.x, b.x), std::min(a.y, b.y));
}

/// \relates Vector2
/// Returns a vector with components which are the maximum respective components
/// of the two vectors
inline Vector2 Max(const Vector2 &a, const Vector2 &b) {
    return Vector2(std::max(a.x, b.x), std::max(a.y, b.y));
}

/// Two dimensional axis alligned bounding box.
struct Box2 {
    Vector2 pmin; ///< The lower bounds for the box along the two axes
    Vector2 pmax; ///< The lower bounds for the box along the two axes

    Box2() {}
    Box2(const Vector2 &min, const Vector2 &max) : pmin(min), pmax(max) {}

    /// Initializes the box to an empty one
    void init() {
        pmin.set(1e20f, 1e20f);
        pmax.set(-1e20f, -1e20f);
    }

    /// Initializes the box with the given extents
    void set(const Vector2 &min, const Vector2 &max) {
        pmin = min;
        pmax = max;
    }

    ///< Returns the size of the box along the x axis
    float dx() const {
        return pmax.x - pmin.x;
    }

    ///< Returns the size of the box along the y axis
    float dy() const {
        return pmax.y - pmin.y;
    }

    /// Compute the area of the box.
    float getArea() const { return dx() * dy(); }

    /// Expands the box (if necessary) to contain the given point
    void operator+=(const Vector2 &p) {
        pmin = Min(pmin, p);
        pmax = Max(pmax, p);
    }

    /// Return a corner of the box.
    /// @param corner The index (0 to 3) of the corner to return.
    /// @return The position of the corner with the specified index.
    Vector2 operator[](int corner) const {
        const Vector2 *p = &pmin;
        return Vector2(p[corner & 1].x, p[(corner >> 1) & 1].y);
    }
};

/// Intersection of a pixel footprint with a box.
/// @note Intentionally, the case where the footprint is inside the box is not
/// handled and instead BoxIntersectionTest_boundary is returned.
enum BoxIntersectionTest {
    boxIntersectionTest_disjoint, ///< the footprint and the box are disjoint
    boxIntersectionTest_boundary, ///< the footprint and the box boundaries intersect
    boxIntersectionTest_inside,   ///< the box is inside the footprint
};

/// Parallelogram approximatioln of a pixel footprint.
class PixelFootprint {
    Vector2 v[4];  // the four vertices
    Vector2 n[2];  // the normals to the two edges
    Box2 bbox;     // the bounding box of the parallelogram
    float area; // the area of the parallelogram

public:
    /// Initialize the parallelogram with a point and two edges.
    /// @param p point in 2d
    /// @param e0 first edge, pointing away from p
    /// @param e1 second edge, pointing away from p
    /// @note box, normals to the edges and the area are computed
    void init(const Vector2 &p, const Vector2 &e0, const Vector2 &e1) {
        Vector2 edge0 = e0;
        Vector2 edge1 = e1;

        area = crossf(e0, e1);
        if (area > 0.0f) {
            std::swap(edge0, edge1);
        } else {
            area = -area;
        }

        n[0] = Vector2(-edge0.y, edge0.x);
        n[1] = Vector2(-edge1.y, edge1.x);

        v[0] = p + edge0;
        v[1] = p;
        v[2] = p + edge1;
        v[3] = v[2] + edge0;

        bbox.init();
        for (int i = 0; i < 4; i++) {
            bbox += v[i];
        }
    }

    /// Get parallelogram corner.
    /// @param i index of the corner between 0 and 3
    const Vector2 &operator[](const int i) const { return v[i]; }

    /// Translate the parallelogram with a given offset.
    /// @param offset translation offset.
    void translate(Vector2 offset) {
        v[0] += offset;
        v[1] += offset;
        v[2] += offset;
        v[3] += offset;

        bbox.pmin += offset;
        bbox.pmax += offset;
    }

    /// Get the first edge of the parallelogram.
    Vector2 getEdge0() const { return v[0] - v[1]; }

    /// Get the second edge of the parallelogram.
    Vector2 getEdge1() const { return v[2] - v[1]; }

    /// Get the bounding box of the parallelogram.
    Box2 getBBox() const { return bbox; }

    /// Get the parallelogram's area.
    float getArea() const { return area; }

    /// Get the parallelogaram area in the unit square.
    float getUnitArea() const {
        // If the footprint in entirely inside the unit square just return its area.
        if (bbox.pmin.x >= 0.0f && bbox.pmin.y >= 0.0f && bbox.pmax.x <= 1.0f && bbox.pmax.y <= 1.0f)
            return area;

        Box2 unitSquare;
        unitSquare.set(Vector2(0.0f, 0.0f), Vector2(1.0f, 1.0f));
        const float unitArea = computeOverlappingExact(unitSquare);
        return unitArea;
    }

    /// Get a uniform random point in the parallelogram, given a uniform random point in the unit square.
    Vector2 samplePoint(Vector2 uv) const {
        Vector2 p = v[1];
        p += getEdge0() * uv.x;
        p += getEdge1() * uv.y;
        return p;
    }

    /// Check whether the point p is inside the parallelogram. This calculation is more efficient than two point-triangle tests.
    int isInside(const Vector2 &p) const {
        Vector2 pos = p - v[1]; // the position of p, relative to v[1]
        // Perform four line point sidedness checks to answer whether the quiery point is inside the parallelogram.
        if (dotf(n[0], pos) < 0.0f && dotf(n[1], pos) > 0.0f) {
            pos = v[3] - p; // the position of p, relative to v[3]
            if (dotf(n[0], pos) < 0.0 && dotf(n[1], pos) > 0.0f) {
                return true;
            }
        }

        return false;
    }

    /// Box-parallelogram intersection test.
    /// Based on a separation line algorithm for intersection tests between
    /// polygons. Optimized for the particular case of AABB and parallelogram.
    /// Additionally the important case, when the box is inside the
    /// parallelogram is handled.
    /// @param b the box to test
    /// @retval one of BoxIntersectionTest_disjoint,
    /// BoxIntersectionTest_boundary and BoxIntersectionTest_inside.
    /// @note Intentionally, the case where the primitive is inside the box is
    /// not handled and instead BoxIntersectionTest_boundary is returned.
    inline BoxIntersectionTest intersectionTest(const Box2 &b) const {

        // Check for separation lines, according to the box axises.
        // We are interested in area intersections, therefore, contour
        // intersections are irrelevant for us.
        if (b.pmin.x >= bbox.pmax.x || b.pmax.x <= bbox.pmin.x || b.pmin.y >= bbox.pmax.y || b.pmax.y <= bbox.pmin.y) {
            return boxIntersectionTest_disjoint;
        }

        // Check for separation lines, according to the parallelogram edges.
        float bmin[2], bmax[2]; // the min and max projections of the box onto the parallelogram normals
        float pmin[2], pmax[2]; // the min and max projections of the parallelogram onto the parallelogram normals
        for (int i = 0; i < 2; i++) { // loop over the two parallelogram normals
            Vector2 norm = n[i];

            bmin[i] = bmax[i] = dotf(norm, b[0]);
            for (int j = 1; j < 4; j++) {
                float proj = dotf(norm, b[j]);
                if (proj < bmin[i])
                    bmin[i] = proj;
                else if (proj > bmax[i])
                    bmax[i] = proj;
            }

            pmin[i] = pmax[i] = dotf(norm, v[i + 1]);
            float proj        = dotf(norm, v[i + 2]);
            if (proj < pmin[i])
                pmin[i] = proj;
            else if (proj > pmax[i])
                pmax[i] = proj;

            if (bmin[i] >= pmax[i] || bmax[i] <= pmin[i]) {
                return boxIntersectionTest_disjoint;
            }
        }

        // If the box projections lie in between of the parallelogram
        // projections, for both parallelogram normals, then the box is entirely
        // inside the parallelogram.
        if (bmin[0] >= pmin[0] && bmax[0] <= pmax[0] && bmin[1] >= pmin[1] && bmax[1] <= pmax[1]) {
            return boxIntersectionTest_inside;
        } else { // Otherwise their edges intersect.
            return boxIntersectionTest_boundary;
        }
    }

    /// Compute intersection of the line between p0 and p1 and axis parallel
    /// line, intersecting the other axis at clip.
    /// @param p0 the starting point of the line to check
    /// @param p1 the ending point of the line to check
    /// @param clip the position of theaxis parallel clipping line
    /// @param compIdx 0 if the clipping line is parallel to x, 1 if parallel to y
    /// @param[out] res the intersection point along p0-p1 line
    /// @retval true if there is intersection, false otherwise
    bool intersectLine(const Vector2 &p0, const Vector2 &p1, float clip, int compIdx, Vector2 &res) const {
        Vector2 d = p1 - p0;
        float t = (clip - p0[compIdx])/d[compIdx]; // division by zero is possible here

        if (enoki::isfinite(t)) { // handle NaNs, if division by zero happened
            int comp2Idx  = !compIdx;
            res[comp2Idx] = p0[comp2Idx] + d[comp2Idx] * t;
            res[compIdx]  = clip;

            return true;
        }

        return false;
    }

    /// Compute the overlapping area of the parallelogram and a box.
    /// Based on Sutherland-Hodgman algorithm, and tuned for the particular case
    /// of box and parallelogram. The box is the clipping convex polygon, and
    /// the parallelogram is the clipped polygon.
    /// @param b the box
    /// @retval The area of the intersection.
    float computeOverlappingExact(const Box2 &b) const {

        Vector2 verts[16];
        int numVerts;

        // We translate both figures closer to the origin because crossf() determinants
        // could be inaccurate for small polygons further away from the origin.
        // When the two figures have very different sizes it is important that the smaller is closer to the origin.
        const Vector2 translation = (area > b.getArea()) ? b.pmin : v[0];

        numVerts = 4;
        for (int i = 0; i < 4; i++) {
            verts[i] = v[i] - translation;
        }
        Box2 transBox = Box2(b.pmin - translation, b.pmax - translation);

        int numClipped;
        Vector2 clippedPoly[16];

        const Vector2 *ps = &transBox.pmin; // box points
        for (int i = 0; i < 4;
             i++) { // loop over the clipping lines (box sides)
            int vectIdx = (i & 2) >> 1; // 0 for pmin and 1 for pmax vector
            int compIdx = i & 1;        // 0 for x and 1 for y component

            float clip = ps[vectIdx][compIdx];
            numClipped = 0;

            for (int j = 0; j < numVerts;
                 j++) { // loop over the parallelogram edges
                Vector2 p0 = verts[j];
                Vector2 p1 = verts[(j + 1) % numVerts];

                // For vectIdx==0, the inequality must be inverted. Therefore,
                // XNOR (<check>==vectIdx) is used.
                if ((p0[compIdx] < clip) == vectIdx) {
                    if ((p1[compIdx] < clip) == vectIdx) {
                        clippedPoly[numClipped++] = p1;
                    } else {
                        Vector2 intersection;
                        if (intersectLine(p0, p1, clip, compIdx, intersection)) {
                            clippedPoly[numClipped++] = intersection;
                        }
                    }
                } else {
                    if ((p1[compIdx] < clip) == vectIdx) {
                        Vector2 intersection;
                        if (intersectLine(p0, p1, clip, compIdx, intersection)) {
                            clippedPoly[numClipped++] = intersection;
                            clippedPoly[numClipped++] = p1;
                        }
                    }
                }
            }

            assert(numClipped <= 15);

            numVerts = numClipped;
            for (int j = 0; j < numClipped; j++) {
                verts[j] = clippedPoly[j];
            }
        }

        // Compute the doubled area by summing the determinants.
        float doubledArea = 0.0f;
        for (int i = 0; i < numVerts; i++) {
            doubledArea += crossf(verts[i], verts[(i + 1) % numVerts]);
        }

        return 0.5f * doubledArea;
    }
};

using Float = float;
using Vector3f = Vector<Float, 3>;
using Normal3f = Normal<Float, 3>;
using Point2f  = Point<Float, 2>;
using Frame3f  = Frame<Float>;
using Color3f  = Color<Float, 3>;

/// A point with two uint16 coordinate for normal map indexing.
struct Point16 {
    union {
        struct {
            uint16_t i;
            uint16_t j;
        };
        uint16_t p[2];
    };

    /// Returns the i-th component (0 for x, 1 for y, 2 for z)
    uint16_t &operator[](const int index) { return p[index]; }

    /// Returns the i-th component (0 for x, 1 for y, 2 for z) as a const
    const uint16_t &operator[](const int index) const { return p[index]; }

    /// Compare operator for sorting.
    bool operator<(const Point16 &p) const {
        return i == p.i ? j < p.j : i < p.i;
    }
};

Normal3f getNormal(float* mapData, int i, int j, int width) {
    const float nx = 2.0f * mapData[3 * (j * width + i)] - 1.0f;
    const float ny = 2.0f * mapData[3 * (j * width + i) + 1] - 1.0f;
    const float nz = 2.0f * mapData[3 * (j * width + i) + 2] - 1.0f;
    return Normal3f(nx, ny, nz);
}

Normal3f sampleMap(const Bitmap& map, float u, float v) {
    const int width = map.width();
    const int height = map.height();

    float fx = u * float(width)+0.5f;
    float fy = v * float(height)+0.5f;

    int x1 = int(fx); 
    fx -= float(x1);
    if (x1 >= width) x1 = 0;
    int y1 = int(fy); 
    fy -= float(y1);
    if (y1 >= height) y1 = 0;

    int x0 = x1 - 1;
    if (x0 < 0) x0 = width - 1;
    int y0 = y1 - 1;
    if (y0 < 0) y0 = height - 1;

    float* mapData = (float*)map.data();

    Normal3f n00 = getNormal(mapData, x0, y0, width);
    Normal3f n10 = getNormal(mapData, x1, y0, width);
    Normal3f n11 = getNormal(mapData, x1, y1, width);
    Normal3f n01 = getNormal(mapData, x0, y1, width);

    Normal3f n0 = (1.0f-fx)*n00 + fx * n10;
    Normal3f n1 = (1.0f-fx)*n01 + fx * n11;
    Normal3f n = (1.0f-fy)*n0 + fy * n1;
    return n;
}

/// Class for storing a normal map in convenient form for evaluation and
/// sampling inside a given pixel footprint, based on high resolution binning.
class NDF {
    int width, height; ///> The size of the normal map.
    float invWidth, invHeight; ///> 1/width and 1/height values.
    int binSubdivs; ///> Square root of the bin resolution.
    int leafSize; ///> The maximum number of elements in a leaf of the constructed bin trees.
    int filterMaps; ///< True if the filtering is enabled and false otherwise.

    std::vector<int> binMap; ///> Map with the bin indices instead of the original normals.
    std::vector<Point16> inverseBinMap; ///> Per bin lists with texel indices into the bin map.
    std::vector<int> forest; ///> Splitting positions for the trees built to the large (>leafSize) bins.
    std::vector<std::pair<int, int>> binLocation; ///> Mapping bin index to a pair of offset (first) and list size (second). The offset is in forest topology for large lists and inverse bin map for short ones.

public:
    NDF() { reset(); }

    ~NDF() { freeMem(); }

    void reset() {
        width = height = 0;
        binSubdivs = 0;
        leafSize = 16;
        filterMaps = true;

        binMap.clear();
        inverseBinMap.clear();
        forest.clear();
        binLocation.clear();
    }

    void freeMem() { reset(); }

    /// Initialize the NDF.
    /// @param map - the normal map to build the structure from.
    /// @param bumpAmount - an interpolation parameter that could draw each
    /// normal closer to (0, 0, 1). Value 1 means that normals are unchanged,
    /// while value 0 moves all normals to (0, 0, 1).
    /// @param binSubdivs - square root of the bin resolution.
    /// @param leafSize - the maximum number of elements in a leaf of the
    /// constructed bin trees.
    /// @param filterMaps - True if filtering is enabled and false otherwise.
    /// @retval true on successful initialization and false otherwise.
    int init(const Bitmap &map, float bumpAmount, int binSubdivs, int leafSize, int filterMaps, int samplingRate) {
        width  = samplingRate*map.width();
        height = samplingRate*map.height();
        if (width > 0 && height > 0) {
            this->invWidth = 1.0f/float(width);
            this->invHeight = 1.0f/float(height);
            this->binSubdivs = binSubdivs;
            this->leafSize = leafSize;
            this->filterMaps = filterMaps;

            binMap.resize(width*height, -1);
            binLocation.resize(binSubdivs*binSubdivs, std::pair<int, int>(0, 0));

            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    Normal3f n = sampleMap(map, (float(i)+0.5f) / float(width), (float(j)+0.5f)/float(height));

                    const float t = bumpAmount;
                    n = normalize(t*n + (1.0f-t)*Normal3f(0.0f, 0.0f, 1.0f));

                    int binIndex = -1;
                    if (n.z() > 0.0f) {
                        float fx, fy;
                        fx = fy = 0.0f;
                        binIndex = encode(n, fx, fy);

                        if (filterMaps) {
                            std::pair<int, int>& loc = binLocation[binIndex];
                            loc.second++;
                        }

                        // Initialize the bin map.
                        binMap[j*width + i] = binIndex;
                    }
                }
            }

            // If filtering is enabled build the inverse bin map and the bin trees.
            if (filterMaps) {
                buildInverseBinMap();
                buildForest();
                assert(testInverseMappingTopology() == 0);
            }

            return true;
        }
        return false;
    }

    // Get memory usage in MB.
    int getMemoryUsage() const {
        uint64_t mem = sizeof(NDF);
        mem += binMap.capacity() * sizeof(int);
        mem += inverseBinMap.capacity() * sizeof(Point16);
        mem += forest.capacity() * sizeof(int);
        mem += binLocation.capacity() * sizeof(std::pair<int, int>);
        return int(mem>>20);
    }

    /// Get the normal at location (i, j).
    /// @note If the normal map is stored, then the orignilal normal is
    /// returned. If not, then the corresponding bin center is returned.
    Vector3f getNormal(int i, int j) const {
        return decode(binMap[j*width + i], 0.5f, 0.5f);
    }

    /// Get the normal for a given point in the unit square.
    Vector3f getNormal(Vector2 p) const {
        const int i = clamp(int(p.x * float(width)), 0, width - 1);
        const int j = clamp(int(p.y * float(height)), 0, height - 1);
        return getNormal(i, j);
    }

    /// Get the bin index at location (i, j).
    int getBinIndex(int i, int j) const { return binMap[j*width + i]; }

    /// Get the bin index for a given point in the unit square.
    int getBinIndex(Vector2 p) const {
        const int i = clamp(int(p.x * float(width)), 0, width - 1);
        const int j = clamp(int(p.y * float(height)), 0, height - 1);
        return getBinIndex(i, j);
    }

    //***************************************************************
    // Evaluation and sampling
    //***************************************************************
    /// Evaluate what portion of the footprint is occupied by the given bin.
    /// @param footprint - the pixel footprint in the unit square.
    /// @param binIndex - the index of the bin to be queried.
    /// @retval A number in the range [0, 1] representing what portion of the
    /// footprint is covered by the bin.
    /// @note When the footprint is partially outside the unit square, then we
    /// search only inside and divide by the footprint area inside the unit square.
    float evalBinCoverage(const PixelFootprint &footprint, int binIndex) const {

        if (!filterMaps) {
            Vector2 p     = footprint.samplePoint(Vector2(0.5f, 0.5f));
            const int idx = getBinIndex(p);

            // If filtering is disabled we consider only the texel at the hit.
            if (binIndex == idx) {
                return 1.0f;
            } else {
                return 0.0f;
            }
        }

        std::pair<int, int> loc = binLocation[binIndex];
        if (loc.second == 0) {
            return 0.0f;
        }

        const int offset  = loc.first;
        const int binSize = loc.second;
        const float contrib = evalContributionTraversal(offset, binSize, footprint);

        if (contrib == 0.0f) {
            return 0.0f;
        }

        const float area = footprint.getUnitArea();
        const float res  = contrib/area;

        return res;
    }

    void evalBinContributionBeckmann(float projArea, Vector3f n, Vector3f h, Vector3f v, Vector3f l, float beckmannAlpha, float &contrib, float &prob) const {

        const Frame3f localFrame(n);
        const Vector3f lh = localFrame.to_local(h);
        const Vector3f lv = localFrame.to_local(v);
        const Vector3f ll = localFrame.to_local(l);

        const float ch = localFrame.cos_theta(lh);
        const float cv = localFrame.cos_theta(lv);
        const float cl = localFrame.cos_theta(ll);

        if (ch > 0.0f && cv > 0.0f && cl > 0.0f) {
            MicrofacetDistribution<Float, Color3f> distBeckmann(MicrofacetType::Beckmann, beckmannAlpha, false);
            const float d = distBeckmann.eval(lh);
            const float g = distBeckmann.G(lv, ll, lh);

            contrib += d*g*projArea;
            prob += d*projArea;
        }
    }

    /// Direct evaluation of all texels under the footprint using a Beckmann lobe.
    /// @param footprint - the pixel footprint in the unit square.
    /// @param h - the half vector.
    /// @retval - the Beckmann lobe evaluated for all the texel normals under
    /// the footprint respecting the texels' areas.
    float evalDirectBeckmann(const PixelFootprint &footprint, Vector3f h, Vector3f v, Vector3f l, float beckmannAlpha, float &prob) const {

        int i0, i1, j0, j1;
        if (filterMaps) {
            const Box2 b = footprint.getBBox();
            i0 = clamp(floor(float(width) * b.pmin.x), 0, width - 1);
            i1 = clamp(ceil(float(width) * b.pmax.x), 0, width);
            j0 = clamp(floor(float(height) * b.pmin.y), 0, height - 1);
            j1 = clamp(ceil(float(height) * b.pmax.y), 0, height);
        } else {
            const Vector2 c = footprint.samplePoint(Vector2(0.5f, 0.5f));
            i0 = clamp(floor(float(width) * c.x), 0, width - 1);
            i1 = i0 + 1;
            j0 = clamp(floor(float(height) * c.y), 0, height - 1);
            j1 = j0 + 1;
        }

        // Find the bin corresponding to the half vector.
        float fx = 0.0f;
        float fy = 0.0f;
        const int binIndex = encode(h, fx, fy);
        const int xh = binIndex % binSubdivs;
        const int yh = binIndex / binSubdivs;

        float contrib = 0.0f;
        for (int j = j0; j < j1; j++) {
            for (int i = i0; i < i1; i++) {
                // Check if the current bin is within the Beckmann filter
                // radius. This is an important optimization since the
                // computation of "texelArea" is expensive.
                const int bin = getBinIndex(i, j);
                const int x = bin % binSubdivs;
                const int y = bin / binSubdivs;
                if (float(sqr(x - xh) + sqr(y - yh)) > sqr(0.5 * float(FILTER_SIZE))) continue;

                float texelArea = 1.0f;
                if (filterMaps) {
                    Point16 p;
                    p.i = i;
                    p.j = j;
                    texelArea = computeTexelOverlappingExact(footprint, p);
                }

                if (texelArea > 0.0f) {
                    const Vector normal = getNormal(i, j);
                    if (normal.z() > 0.0f) {
                        evalBinContributionBeckmann(texelArea, normal, h, v, l, beckmannAlpha, contrib, prob);
                    }
                }
            }
        }

        if (filterMaps) {
            const float unitArea = footprint.getUnitArea();
            contrib /= unitArea;
            prob /= unitArea;
        }

        return contrib;
    }

    const static int FILTER_RADIUS = 2; // Number of adjacent bins of the half vector bin in each direction. Tested with 2 and 3.
    const static int FILTER_SIZE = 2 * FILTER_RADIUS + 1; // The width/height of all queried bins.
    // Size for batch queries (total number of bins for query).
    // Note that the four corners of the rectangle FILTER_SIZE*FILTER_SIZE are cut to resemble a circle within 3 sigma.
    const static int BATCH_SIZE = FILTER_SIZE * FILTER_SIZE - 4 * (FILTER_RADIUS == 2 ? 1 : 3);

    float evalFilteredBeckmannBatch(const PixelFootprint &footprint, Vector3f h, Vector3f v, Vector3f l, float beckmannAlpha, float &prob) const {

        const float footprintTexels = footprint.getBBox().getArea() * width * height;
        if (!filterMaps || footprintTexels < sqr(8.0f)) {
            return evalDirectBeckmann(footprint, h, v, l, beckmannAlpha, prob);
        }

        struct BinArea {
            int idx;
            float area;
        };
        BinArea binAreas[BATCH_SIZE];
        int numActiveBins = 0;

        int numTrees = 0;
        int bins[BATCH_SIZE];
        int offsets[BATCH_SIZE];
        int sizes[BATCH_SIZE];
        float contribs[BATCH_SIZE];

        float fx = 0.0f;
        float fy = 0.0f;
        const int binIndex = encode(h, fx, fy);
        const int xh = binIndex % binSubdivs;
        const int yh = binIndex / binSubdivs;

        const int x0 = std::max(xh - FILTER_RADIUS, 0);
        const int x1 = std::min(xh + FILTER_RADIUS, binSubdivs - 1);
        const int y0 = std::max(yh - FILTER_RADIUS, 0);
        const int y1 = std::min(yh + FILTER_RADIUS, binSubdivs - 1);

        for (int j = y0; j <= y1; j++) {
            const int s = std::max(abs(y1 - j - FILTER_RADIUS) - 1, 0);
            for (int i = x0 + s; i <= x1 - s; i++) {
                const int bin = j * binSubdivs + i;
                std::pair<int, int> loc = binLocation[bin];

                if (loc.second != 0) {
                    const int offset = loc.first;
                    const int size = loc.second;

                    if (size <= leafSize) {
                        const float area = evalContributionLinear(offset, size, footprint);

                        if (area > 0.0f) {
                            BinArea &b = binAreas[numActiveBins++];
                            b.idx = bin;
                            b.area = area;
                        }
                    } else {
                        bins[numTrees] = bin;
                        offsets[numTrees] = offset;
                        sizes[numTrees] = size;
                        numTrees++;
                    }
                }
            }
        }

        evalContributionTraversal(offsets, sizes, contribs, numTrees, footprint);

        for (int c = 0; c < numTrees; c++) {
            if (contribs[c] > 0.0f) {
                BinArea &b = binAreas[numActiveBins++];
                b.idx = bins[c];
                b.area = contribs[c];
            }
        }

        const float footprintArea = footprint.getUnitArea();

        float contrib = 0.0f;
        for (int i = 0; i < numActiveBins; i++) {
            const BinArea &binArea = binAreas[i];
            const Vector normal = decode(binArea.idx, 0.5f, 0.5f);
            if (normal.z() > 0.0f) {
                const float binCoverage = binArea.area/footprintArea;
                evalBinContributionBeckmann(binCoverage, normal, h, v, l, beckmannAlpha, contrib, prob);
            }
        }

        return contrib;
    }

    /// Sample new direction based on the NDF. Random point is sampled inside
    /// the footprint to find a random footprint texel. Then a random normal is
    /// sampled from a Beckmann distribution. Then the view direction is
    /// reflected from it.
    /// @param footprint - the pixel footprint to sample from.
    /// @param u - the first random variable.
    /// @param v - the second random variable.
    /// @param localViewDir - the view direction in shading space.
    /// @param uvwToShade - a matrix that transforms normals from UVW space to
    /// shading space.
    /// @param alpha - the Beckmann roughness parameter.
    /// @retval the new sampled direction in shading space.
    /// @note If the sampled point happens to be outside the unit square, than
    /// the point and the footprint are translated to the unit square.
    Vector3f sampleBeckmann(PixelFootprint &footprint, const Point2f &sample1, const Point2f &sample2, Vector3f localViewDir, float alpha) const {

        Normal3f localMicronormal(0.0f);

        Vector2 p;
        if (filterMaps) {
            p = footprint.samplePoint(Vector2(sample1.x(), sample1.y()));
        } else {
            p = footprint.samplePoint(Vector2(0.5f, 0.5f));
        }

        // Move the point to the unit square.
        // This is related to the unit area in the eval method.
        Vector2 fp = Vector2(p.x - floor(p.x), p.y - floor(p.y));

        int idx = getBinIndex(fp);
        if (idx != -1) {
            const Vector3f normal = getNormal(fp);

            MicrofacetDistribution<Float, Color3f> dist(MicrofacetType::Beckmann, alpha, false);
            const std::pair<Normal3f, Float> bnorm = dist.sample(localViewDir, sample2);

            const Frame3f localFrame(normal);
            localMicronormal = localFrame.to_world(bnorm.first);
        }

        // Translate the footprint to its correct location in the unit square.
        footprint.translate(fp - p);
        p = fp;

        return localMicronormal;
    }

private:
    //***************************************************************
    // Building and queries
    //***************************************************************

    // Loop through all bin entries and accumulate the proper
    // bin offsets. Return the number of nonempty bins.
    int updateBinOffsets() {
        int offset = 0;
        for (int i = 0; i < binSubdivs * binSubdivs; i++) {
            std::pair<int, int> &loc = binLocation[i];

            if (loc.second > 0) {
                loc.first = offset;
                offset += loc.second;
            }
        }
        return offset;
    }

    // Build the inverse bin map.
    void buildInverseBinMap() {
        // First the correct bin offsets are computed.
        const int validTexels = updateBinOffsets();

        // Then the inverse bin map is initialized.
        // During this process the offsets are destroyed, because they are used
        // to store the location of the next free slot in the corresponding bin
        // list.
        inverseBinMap.resize(validTexels);

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                const int binIdx = binMap[j * width + i];
                if (binIdx != -1) {
                    std::pair<int, int> &loc = binLocation[binIdx];
                    if (loc.second != 0) {
                        int &offset = loc.first;
                        Point16 &p = inverseBinMap[offset];
                        p.i = i;
                        p.j = j;
                        offset++;
                    }
                }
            }
        }

        // Restore the original offsets.
        for (int i = 0; i < binSubdivs * binSubdivs; i++) {
            std::pair<int, int> &loc = binLocation[i];
            loc.first -= loc.second;
        }
    }

    // Compute an approximation of the area of the intersection between the
    // texel p and the footprint. if the texel center is inside the footprint we
    // assume that the whole is inside.
    float computeTexelOverlappingApprox(const PixelFootprint &footprint, const Point16 &p) const {
        Vector2 vp;
        vp.x = (float(p.i) + 0.5f) * invWidth;
        vp.y = (float(p.j) + 0.5f) * invHeight;

        if (footprint.isInside(vp)) {
            return invWidth * invHeight;
        }
        return 0.0f;
    }

    // Compute the area of the intersection of the texel p and the footprint.
    float computeTexelOverlappingExact(const PixelFootprint &footprint, const Point16 &p) const {
        Vector2 pmin(float(p.i) * invWidth, float(p.j) * invHeight);
        Vector2 pmax(float(p.i + 1.0f) * invWidth, float(p.j + 1.0f) * invHeight);

        const Box2 footprintBox = footprint.getBBox();

        // Check for separation lines, according to the box axises.
        // We are interested in area intersections, therefore, contour
        // intersections are irrelevant for us.
        bool disjoint = (pmin.x >= footprintBox.pmax.x) | (pmin.y >= footprintBox.pmax.y) | (pmax.x <= footprintBox.pmin.x) | (pmax.y <= footprintBox.pmin.y);

        float area = 0.0f;
        if (!disjoint) {
            Box2 texel;
            texel.set(pmin, pmax);
            area = footprint.computeOverlappingExact(texel);
        }

        return area;
    }

    // Contribution evaluation strategy - linear loop through all elements in a
    // given bin. This is used for small bins and for tree leaves, and it is a
    // simple strategy for testing as well.
    float evalContributionLinear(const int binOffset, const int binSize, const PixelFootprint &footprint) const {
        float contrib = 0.0f;
        const Point16 *pos = inverseBinMap.data() + binOffset;

        for (int j = 0; j < binSize; j++) {
            const Point16 p = pos[j];

            const float currentContrib = computeTexelOverlappingApprox(footprint, p);
            contrib += currentContrib;
        }
        return contrib;
    }

    // Build all binary trees for the bins larger than leafSize.
    // Single-threaded for now, but can be parallelized.
    void buildForest() {

        // Build a tree for each bin with size larger than leafSize.
        for (int i = 0; i < binSubdivs * binSubdivs; i++) {
            std::pair<int, int> &loc = binLocation[i];
            const int offset = loc.first;
            const int size = loc.second;

            if (size > leafSize) {
                loc.first = forest.size();
                forest.push_back(offset);

                buildTree(offset, offset + size);
            }
        }
    }

    // The maximum stack size for building and traversal.
    const static int STACK_SIZE = 32;
    // A binary tree node used both for tree building and for traversal.
    // The node store data for N bins. N>1 is used for batch queries.
    template <int N> struct Node {
        int left[N];  // Index to the start position of this node into the inverse bin map.
        int right[N]; // Index to the end position of this node into the inverse bin map.
        int splitOffset[N]; // Index of the splitting element.
        uint16_t dim[2];    // The width and height of the node.
        uint16_t offset[2]; // The offsets of the node from (0, 0).

        Node() {}

        // Split the node into two children. Always the larger side is split
        // into to halves. The split position is unknown so the left index of
        // the second and the right index of the first are no set.
        int getChildren(Node *children) const {
            std::memset(&children[0].splitOffset, -1, sizeof(int) * N);
            std::memset(&children[1].splitOffset, -1, sizeof(int) * N);

            std::memcpy(&children[0].left, &left, sizeof(int) * N);
            std::memcpy(&children[1].right, &right, sizeof(int) * N);

            const int dimIdx = dim[0] >= dim[1] ? 0 : 1;

            children[0].dim[0] = dim[0];
            children[0].dim[1] = dim[1];
            children[0].dim[dimIdx] /= 2;

            children[1].dim[0] = children[0].dim[0];
            children[1].dim[1] = children[0].dim[1];

            const int remainder = dim[dimIdx] % 2;
            children[1].dim[dimIdx] += remainder;

            children[0].offset[0] = children[1].offset[0] = offset[0];
            children[0].offset[1] = children[1].offset[1] = offset[1];
            children[1].offset[dimIdx] += children[0].dim[dimIdx];

            return dimIdx;
        }

        // Number of element in the node.
        int numElements(int i = 0) const { return right[i] - left[i]; }

        // Compute the node representation in the uni square.
        Box2 getUnitBox(float invWidth, float invHeight) const {
            Box2 res;
            res.pmin.x = float(offset[0]) * invWidth;
            res.pmin.y = float(offset[1]) * invHeight;
            res.pmax.x = float(offset[0] + dim[0]) * invWidth;
            res.pmax.y = float(offset[1] + dim[1]) * invHeight;
            return res;
        }
    };

    /// Order a range of the inverse bin map so that first portion is smaller
    /// than "pivot".
    ///@param left The start index of the range to partition.
    ///@param right The end index of the range to partition.
    ///@param pivot The pivot for the ordering.
    ///@param dim The dimension along which the ordering done.
    ///@retval The index "res" which divides the "pos" array.
    ///@note Resulting elements [left, res) are stricktly smaller than "pivot"
    /// and elements [res, right) are greater or equal.
    int partition(int left, int right, int pivot, int dim) {

        int i = left;
        int j = right - 1;
        while (true) {
            while (i <= j && inverseBinMap[i][dim] < pivot)
                i++;
            while (pivot <= inverseBinMap[j][dim] && i < j)
                j--;

            if (i < j) {
                std::swap(inverseBinMap[i], inverseBinMap[j]);
                i++;
            } else {
                break;
            }
        }

        assert(testPartition(left, right, pivot, dim, i) == 0);

        return i;
    }

    // Given a range [left, right) in the inverse bin map, build a tree for
    // evaluation acceleration.
    void buildTree(int left, int right) {

        Node<1> nodes[STACK_SIZE];

        int stackTop = 0;
        Node<1> &root = nodes[stackTop++];
        root.left[0] = left;
        root.right[0] = right;
        root.splitOffset[0] = -1;
        root.dim[0] = width;
        root.dim[1] = height;
        root.offset[0] = 0;
        root.offset[1] = 0;

        while (stackTop) {
            Node<1> node = nodes[--stackTop];

            Node<1> children[2];
            const int dimIdx = node.getChildren(children);
            const int split = partition(node.left[0], node.right[0],
                                        children[1].offset[dimIdx], dimIdx);
            children[0].right[0] = split;
            children[1].left[0] = split;

            if (node.splitOffset[0] != -1) {
                forest[node.splitOffset[0]] = forest.size();
            }
            forest.push_back(split);

            const int firstChildIndex = stackTop;
            for (int i = 0; i < 2; i++) {
                if (children[i].numElements() > leafSize) {
                    nodes[stackTop++] = children[i];
                }
            }

            const int numPushedNodes = stackTop - firstChildIndex;
            if (numPushedNodes == 2) {
                nodes[firstChildIndex].splitOffset[0] = forest.size();
                forest.push_back(-1);
            }

            assert(stackTop >= 0 && stackTop < STACK_SIZE);
        }
    }

    // Contribution evaluation strategy - traverse a tree to find all footprint texels of a given bin.
    float evalContributionTraversal(const int offset, const int binSize, const PixelFootprint &footprint) const {

        float contribution = 0.0f;
        if (binSize <= leafSize) {
            contribution = evalContributionLinear(offset, binSize, footprint);
            return contribution;
        }

        const int left  = forest[offset];
        const int right = left + binSize;

        Node<1> nodes[STACK_SIZE];

        int stackTop = 0;
        Node<1> &root = nodes[stackTop++];
        root.left[0] = left;
        root.right[0] = right;
        root.splitOffset[0] = offset + 1;
        root.dim[0] = width;
        root.dim[1] = height;
        root.offset[0]= 0;
        root.offset[1]= 0;

        while (stackTop) {
            Node<1> node = nodes[--stackTop];

            Node<1> children[2];
            node.getChildren(children);

            const int split = forest[node.splitOffset[0]];
            children[0].right[0] = split;
            children[1].left[0] = split;

            const int childrenElements[2] = { children[0].numElements(),
                                              children[1].numElements() };
            for (int i = 0; i < 2; i++) {
                if (childrenElements[i]) {
                    const Box2 childBox = children[i].getUnitBox(invWidth, invHeight);
                    const BoxIntersectionTest testRes = footprint.intersectionTest(childBox);

                    if (testRes == boxIntersectionTest_inside) {
                        contribution += childrenElements[i] * invWidth * invHeight;
                    } else if (testRes == boxIntersectionTest_boundary) {
                        if (childrenElements[i] > leafSize) { // This child is not a leaf.
                            if (childrenElements[1 - i] >
                                leafSize) { // The other child is also not a leaf.
                                children[i].splitOffset[0] = i ? node.splitOffset[0] + 2 : forest[node.splitOffset[0] + 1];
                            } else {
                                children[i].splitOffset[0] = node.splitOffset[0] + 1;
                            }

                            nodes[stackTop++] = children[i];
                        } else { // Leaf. Compute the contribution of its
                                 // texels.
                            contribution += evalContributionLinear(children[i].left[0], childrenElements[i], footprint);
                        }
                    }
                }
            }

            assert(stackTop >= 0 && stackTop < STACK_SIZE);
        }

        return contribution;
    }

    void evalContributionTraversal(int offset[BATCH_SIZE], int binSize[BATCH_SIZE], float contrib[BATCH_SIZE], int numTrees, const PixelFootprint &footprint) const {

        memset(contrib, 0, sizeof(float) * BATCH_SIZE);

        Node<BATCH_SIZE> nodes[STACK_SIZE];

        int stackTop = 0;
        Node<BATCH_SIZE> &root = nodes[stackTop++];
        root.dim[0] = width;
        root.dim[1] = height;
        root.offset[0] = 0;
        root.offset[1] = 0;

        for (int i = 0; i < numTrees; i++) {
            root.left[i] = forest[offset[i]];
            root.right[i] = root.left[i] + binSize[i];
            root.splitOffset[i] = offset[i] + 1;
        }

        while (stackTop) {
            Node<BATCH_SIZE> node = nodes[--stackTop];

            Node<BATCH_SIZE> children[2];
            node.getChildren(children);

            for (int i = 0; i < numTrees; i++) {
                const int splitOffset = node.splitOffset[i];
                if (splitOffset >= 0) {
                    const int split = forest[splitOffset];
                    children[0].right[i] = split;
                    children[1].left[i] = split;
                } else {
                    children[0].left[i] = 0;
                    children[0].right[i] = 0;
                    children[1].left[i] = 0;
                    children[1].right[i] = 0;
                }
            }

            for (int i = 0; i < 2; i++) {
                const Box2 childBox = children[i].getUnitBox(invWidth, invHeight);
                const BoxIntersectionTest testRes = footprint.intersectionTest(childBox);

                if (testRes == boxIntersectionTest_inside) {
                    for (int c = 0; c < numTrees; c++) {
                        const int childElements = children[i].numElements(c);
                        contrib[c] += childElements * invWidth * invHeight;
                    }
                } else if (testRes == boxIntersectionTest_boundary) {
                    bool pushNode = false;

                    for (int c = 0; c < numTrees; c++) {
                        const int childElements = children[i].numElements(c);
                        if (childElements) {
                            if (childElements > leafSize) { // This child is not a leaf.
                                const int otherChildElements = children[1 - i].numElements(c);
                                if (otherChildElements >
                                    leafSize) { // The other child is also not a leaf.
                                    children[i].splitOffset[c] = i ? node.splitOffset[c] + 2 : forest[node.splitOffset[c] + 1];
                                } else {
                                    children[i].splitOffset[c] = node.splitOffset[c] + 1;
                                }

                                pushNode = true;
                            } else { // Leaf. Compute the contribution of its texels.
                                contrib[c] += evalContributionLinear(children[i].left[c], childElements, footprint);
                            }
                        }
                    }

                    if (pushNode) {
                        nodes[stackTop++] = children[i];
                    }
                }
            }

            assert(stackTop >= 0 && stackTop < STACK_SIZE);
        }
    }

    //***************************************************************************************************************
    // Binning strategy - encode method compute bin index for a given normal
    // and initialize fractional coordinated fx and fy in the bin while decode
    // method compute a normal given bin index and fractional coordinates.
    //***************************************************************************************************************
    int encode(Vector3f n, float& fx, float& fy) const {
        const float u = 0.5f * n.x() + 0.5f;
        const float v = 0.5f * n.y() + 0.5f;

        fx += float(binSubdivs) * u;
        fy += float(binSubdivs) * v;
        int x = int(fx);
        int y = int(fy);
        fx -= float(x);
        fy -= float(y);
        if (x >= binSubdivs) x = binSubdivs - 1;
        if (y >= binSubdivs) y = binSubdivs - 1;

        const int idx = y * binSubdivs + x;
        return idx;
    }

    Vector3f decode(int idx, float fx, float fy) const {
        const int x = idx % binSubdivs;
        const int y = idx / binSubdivs;

        const float u = (float(x) + fx) / float(binSubdivs);
        const float v = (float(y) + fy) / float(binSubdivs);

        Vector3f n;
        n.x() = 2.0f * u - 1.0f;
        n.y() = 2.0f * v - 1.0f;
        const float d = 1.0f - n.x() * n.x() - n.y() * n.y();
        n.z() = d > 0.0f ? sqrtf(d) : 0.0f;
        return n;
    }

    //***************************************************************
    // Testing the data structures
    //***************************************************************

    // Check if a partition is properly computed.
    int testPartition(int left, int right, int pivot, int dim, int res) {
        int errors = 0;
        for (int i = left; i < res; i++) {
            Point16 p = inverseBinMap[i];

            if (p[dim] > pivot) {
                errors++;
            }
        }

        for (int i = res; i < right; i++) {
            Point16 p = inverseBinMap[i];

            if (p[dim] < pivot) {
                errors++;
            }
        }

        return errors;
    }

    // Check if the inverse bin map is properly computed.
    int testInverseMappingTopology() {
        int errorCount = 0;
        for (int j = 0; j < binSubdivs * binSubdivs; j++) {
            std::pair<int, int> loc = binLocation[j];

            int offset = loc.first;
            const int size = loc.second;
            if (size > leafSize) {
                offset = forest[offset];
            }

            for (int i = offset; i < offset + size; i++) {
                Point16 p = inverseBinMap[i];
                int binIdx = binMap[p.j * width + p.i];

                if (binIdx != j) {
                    errorCount++;
                }
            }
        }
        return errorCount;
    }
};

template <typename Float, typename Spectrum>
class MultiscaleMicrofacetBRDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, MicrofacetDistribution)

    MultiscaleMicrofacetBRDF(const Properties &props) : Base(props) {

        if (props.has_property("distribution")) {
            std::string distr = string::to_lower(props.string("distribution"));
            if (distr == "beckmann")
                m_type = MicrofacetType::Beckmann;
            else if (distr == "ggx")
                m_type = MicrofacetType::GGX;
            else
                Throw("Specified an invalid distribution \"%s\", must be "
                    "\"beckmann\" or \"ggx\"!", distr.c_str());
        }
        else {
            m_type = MicrofacetType::Beckmann;
        }

        m_sample_visible = props.bool_("sample_visible", true);

        if (props.has_property("alpha_u") || props.has_property("alpha_v")) {
            if (!props.has_property("alpha_u") || !props.has_property("alpha_v"))
                Throw("Microfacet model: both 'alpha_u' and 'alpha_v' must be specified.");
            if (props.has_property("alpha"))
                Throw("Microfacet model: please specify either 'alpha' or 'alpha_u'/'alpha_v'.");
            m_alpha_u = props.float_("alpha_u");
            m_alpha_v = props.float_("alpha_v");
        } else {
            m_alpha_u = m_alpha_v = props.float_("alpha", 0.1f);
        }

        m_filter_maps  = props.bool_("filter_maps", true);
        m_tiles = props.float_("tiles", 1.f);
        m_scale = props.float_("scale", 1.f);
        m_sampling_rate = props.int_("sampling_rate", 1);

        const float MIN_BECKMANN_ALPHA = 0.0025f;
        m_beckmann_alpha = std::max(props.float_("beckmann_alpha", 0.01f), MIN_BECKMANN_ALPHA);

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_filename = file_path.filename().string();
        Log(Info, "Loading normalmap texture from \"%s\" ..", m_filename);

        Timer timer;
        Log(Info, "Initialize IBM..");
        timer.reset();

        ref<Bitmap> normalmap;
        if (file_path.extension() == ".png" || file_path.extension() == ".jpg") {
            normalmap = new Bitmap(file_path);
            normalmap = normalmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, true);
        } else if (file_path.extension() == ".exr") {
            normalmap = new Bitmap(file_path);
            normalmap = normalmap->convert(Bitmap::PixelFormat::RGB, struct_type_v<ScalarFloat>, false);
        } else {
            Throw("Normalmap(): Unkown file format.");
        }

        if (normalmap.get()) {
            const float sigma = m_beckmann_alpha/sqrtf(2.0f);
            const float theta3 = atanf(3.0f * sigma);
            const int binSubdivs = int(float(NDF::FILTER_SIZE)/sinf(theta3));
            ndf.init(*normalmap, 1.0f /*bumpAmount*/, binSubdivs, 10 /*leafSize*/, m_filter_maps, m_sampling_rate);

            const int memoryUsage = ndf.getMemoryUsage();
            Log(Info, "IBM memory usage: %d MB\n", memoryUsage);
        }

        Log(Info, "done. (took %s)", util::time_string(timer.value(), true));

        parameters_changed({});
    }

    void parameters_changed(const std::vector<std::string> & /*keys*/) override {
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide | BSDFFlags::NeedsDifferentials;
        if (m_alpha_u != m_alpha_v)
            m_flags = m_flags | BSDFFlags::Anisotropic;

        m_components.clear();
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx, const SurfaceInteraction3f &si, Float /*sample1*/, const Point2f &sample2, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            BSDFSample3f bs;
            Float cos_theta_i = Frame3f::cos_theta(si.wi);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || cos_theta_i < 0)) {
                return { bs, 0.f };
            }

            PixelFootprint footprint;

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);

            Sampler<Float, Spectrum> *sampler = (Sampler<Float, Spectrum> *) ctx.sampler;
            Point2f sample1=sampler->next_2d();

            Normal3f m;
            if (!si.has_uv_partials()) {
                Float unused;
                std::tie(m, unused) = distr.sample(si.wi, sample2);
            } else {
                Vector2 uv = Vector2(si.uv.x(), si.uv.y()) * m_tiles;
                uv.x -= floor(uv.x);
                uv.y -= floor(uv.y);
                Vector2 edge0 = Vector2(si.duv_dx[0], si.duv_dx[1]) * m_tiles;
                Vector2 edge1 = Vector2(si.duv_dy[0], si.duv_dy[1]) * m_tiles;
                Vector2 base  = uv - edge0 * 0.5f - edge1 * 0.5f;
                footprint.init(base, edge0, edge1);

                m=ndf.sampleBeckmann(footprint, sample1, sample2, si.wi, m_beckmann_alpha);
            }

            if (m[0] == 0.f && m[1] == 0.f && m[2] == 0.f) {
                return { bs, 0.f };
            }

            bs.wo = reflect(si.wi, m);
            bs.eta = 1.f;
            bs.sampled_component = 0;
            bs.sampled_type =+ BSDFFlags::GlossyReflection;
            if (!si.has_uv_partials()) {
                bs.pdf = pdf(ctx, si, bs.wo, active);
            } else {
                float prob = 0.0f;
                ndf.evalFilteredBeckmannBatch(footprint, m, si.wi, bs.wo, m_beckmann_alpha, prob);
                bs.pdf = prob / (4.f * fabsf(dot(m, bs.wo)));
            }
            if (bs.pdf == 0.f) {
                return { bs, 0.f };
            }

            if (Frame3f::cos_theta(bs.wo) < 0) {
                return { bs, 0.f };
            }

            /* Evaluate Smith's shadow-masking function */
            Float G = distr.G(si.wi, bs.wo, m);

            /* Calculate the total amount of reflection */
            Float model = G / (4.0f * cos_theta_i);
            Spectrum weight = model * (4.0f * fabsf(dot(m, bs.wo))) / Frame3f::cos_theta(m);
            return { bs, weight };
        }
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si, const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return 0.f;
        } else {
            Float cos_theta_i = Frame3f::cos_theta(si.wi), cos_theta_o = Frame3f::cos_theta(wo);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || cos_theta_i < 0.f || cos_theta_o < 0.f)) {
                return 0;
            }

            /* Calculate the half-direction vector */
            Vector3f H = normalize(wo + si.wi);

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);

            Float D = 0.f;
            if (!si.has_uv_partials()) {
                D = distr.eval(H);
            } else {
                Vector2 uv = Vector2(si.uv.x(), si.uv.y()) * m_tiles;
                uv.x -= floor(uv.x);
                uv.y -= floor(uv.y);
                Vector2 edge0 = Vector2(si.duv_dx[0], si.duv_dx[1]) * m_tiles;
                Vector2 edge1 = Vector2(si.duv_dy[0], si.duv_dy[1]) * m_tiles;
                Vector2 base  = uv - edge0 * 0.5f - edge1 * 0.5f;

                PixelFootprint footprint;
                footprint.init(base, edge0, edge1);

                float prob=0.0f;
                D = ndf.evalFilteredBeckmannBatch(footprint, H, si.wi, wo, m_beckmann_alpha, prob);

                // A little hack here. Mitsuba does not give the option to return the probability together with the brdf value while V-Ray does.
                // Since the probability is computed together with the brdf we pass it here and read it in pdf().
                // This works with the path integrator because pdf() is called after eval(). Note that this might not hold for other integrators.
                const_cast<SurfaceInteraction3f &>(si).time = prob;
            }

            if (D == 0.f) {
                return 0.f;
            }

            /* Evaluate Smith's shadow-masking function */
            Float G = distr.G(si.wi, wo, H);

            /* Evaluate the full microfacet model */
            return D * G / (4.f * cos_theta_i);
        }
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si, const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return 0.f;
        } else {
            Float cos_theta_i = Frame3f::cos_theta(si.wi), cos_theta_o = Frame3f::cos_theta(wo);
            if (unlikely(!ctx.is_enabled(BSDFFlags::GlossyReflection) || cos_theta_i < 0.f || cos_theta_o < 0.f)) {
                return 0;
            }

            /* Calculate the half-direction vector */
            Vector3f H = normalize(wo + si.wi);

            /* Construct a microfacet distribution matching the
               roughness values at the current surface position. */
            MicrofacetDistribution distr(m_type, m_alpha_u, m_alpha_v, m_sample_visible);

            Float prob = 0.f;
            if (!si.has_uv_partials()) {
                prob = distr.eval(H) * Frame3f::cos_theta(H);
            } else {
                // This value was computed in eval().
                prob = si.time;
            }

            if (prob == 0.f) {
                return 0.f;
            }

            return prob / (4.f * fabsf(dot(H, wo)));
        }
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultiscaleMicrofacetBRDF[]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    /// Specifies the type of microfacet distribution
    MicrofacetType m_type;
    /// Anisotropic roughness values
    Float m_alpha_u, m_alpha_v;
    /// Importance sample the distribution of visible normals?
    bool m_sample_visible;

    /// Normal/flake map filename
    std::string m_filename;

    bool m_filter_maps;
    Float m_beckmann_alpha;
    Float m_tiles;
    Float m_scale;
    int m_sampling_rate;

    NDF ndf;
};

MTS_IMPLEMENT_CLASS_VARIANT(MultiscaleMicrofacetBRDF, BSDF)
MTS_EXPORT_PLUGIN(MultiscaleMicrofacetBRDF, "Multiscale microfacet model based on inverse bin mapping");

NAMESPACE_END(mitsuba)