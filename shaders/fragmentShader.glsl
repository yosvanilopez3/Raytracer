// set the precision of the float values (necessary if using float)
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
precision mediump int;

// define constant parameters
// EPS is for the precision issue (see precept slide)
#define INFINITY 1.0e+12
#define EPS 1.0e-3

// define constants for scene setting 
#define MAX_LIGHTS 10

// define texture types
#define NONE 0
#define CHECKERBOARD 1
#define MYSPECIAL 2

// define material types
#define BASICMATERIAL 1
#define PHONGMATERIAL 2
#define LAMBERTMATERIAL 3

// define reflect types - how to bounce rays
#define NONEREFLECT 1
#define MIRRORREFLECT 2
#define GLASSREFLECT 3

#define M_PI 3.1415926535897932384626433832795

struct Shape {
    int shapeType;
    vec3 v1;
    vec3 v2;
    float rad;
};

struct Material {
    int materialType;
    vec3 color;
    float shininess;
    vec3 specular;

    int materialReflectType;
    float reflectivity; 
    float refractionRatio;
    int special;

};

struct Object {
    Shape shape;
    Material material;
};

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
    float attenuate;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Intersection {
    vec3 position;
    vec3 normal;
};

// uniform
uniform mat4 uMVMatrix;
uniform int frame;        
uniform float height;
uniform float width;
uniform vec3 camera;
uniform int numObjects;
uniform int numLights;
uniform Light lights[MAX_LIGHTS];
uniform vec3 objectNorm;

varying vec2 v_position;

// find then position some distance along a ray
vec3 rayGetOffset( Ray ray, float dist ) {
    return ray.origin + ( dist * ray.direction );
}

// if a newly found intersection is closer than the best found so far, record the new intersection and return true;
// otherwise leave the best as it was and return false.
bool chooseCloserIntersection(float dist, inout float best_dist, inout Intersection intersect, inout Intersection best_intersect) {
    if (best_dist <= dist) return false;
    best_dist = dist;
    best_intersect.position = intersect.position;
    best_intersect.normal   = intersect.normal;
    return true;
}

// put any general convenience functions you want up here
float triangleArea(vec3 v1, vec3 v2, vec3 v3) {
    float s1Len = distance(v1, v2);
    float s2Len = distance(v2, v3);
    float s3Len = distance(v3, v1); 
    float p     = (s1Len + s2Len + s3Len)/2.0;
    return sqrt(p * (p - s1Len) * (p - s2Len) * (p - s3Len));
}
// psuedo random generator (online implementation, http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl)
float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

/*
    3D Perlin-Noise from example by Stefan Gustavson, found at
    http://staffwww.itn.liu.se/~stegu/simplexnoise/
*/
uniform sampler2D permTexture;          // Permutation texture
const float permTexUnit = 1.0/256.0;        // Perm texture texel-size
const float permTexUnitHalf = 0.5/256.0;    // Half perm texture texel-size
 
float fade(in float t) {
    return t*t*t*(t*(t*6.0-15.0)+10.0);
}
  
float pnoise3D(in vec3 p)
{
    vec3 pi = permTexUnit*floor(p)+permTexUnitHalf; // Integer part, scaled so +1 moves permTexUnit texel
    // and offset 1/2 texel to sample texel centers
    vec3 pf = fract(p);     // Fractional part for interpolation
 
    // Noise contributions from (x=0, y=0), z=0 and z=1
    float perm00 = texture2D(permTexture, pi.xy).a ;
    vec3  grad000 = texture2D(permTexture, vec2(perm00, pi.z)).rgb * 4.0 - 1.0;
    float n000 = dot(grad000, pf);
    vec3  grad001 = texture2D(permTexture, vec2(perm00, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
    float n001 = dot(grad001, pf - vec3(0.0, 0.0, 1.0));
 
    // Noise contributions from (x=0, y=1), z=0 and z=1
    float perm01 = texture2D(permTexture, pi.xy + vec2(0.0, permTexUnit)).a ;
    vec3  grad010 = texture2D(permTexture, vec2(perm01, pi.z)).rgb * 4.0 - 1.0;
    float n010 = dot(grad010, pf - vec3(0.0, 1.0, 0.0));
    vec3  grad011 = texture2D(permTexture, vec2(perm01, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
    float n011 = dot(grad011, pf - vec3(0.0, 1.0, 1.0));
 
    // Noise contributions from (x=1, y=0), z=0 and z=1
    float perm10 = texture2D(permTexture, pi.xy + vec2(permTexUnit, 0.0)).a ;
    vec3  grad100 = texture2D(permTexture, vec2(perm10, pi.z)).rgb * 4.0 - 1.0;
    float n100 = dot(grad100, pf - vec3(1.0, 0.0, 0.0));
    vec3  grad101 = texture2D(permTexture, vec2(perm10, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
    float n101 = dot(grad101, pf - vec3(1.0, 0.0, 1.0));
 
    // Noise contributions from (x=1, y=1), z=0 and z=1
    float perm11 = texture2D(permTexture, pi.xy + vec2(permTexUnit, permTexUnit)).a ;
    vec3  grad110 = texture2D(permTexture, vec2(perm11, pi.z)).rgb * 4.0 - 1.0;
    float n110 = dot(grad110, pf - vec3(1.0, 1.0, 0.0));
    vec3  grad111 = texture2D(permTexture, vec2(perm11, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
    float n111 = dot(grad111, pf - vec3(1.0, 1.0, 1.0));
 
    // Blend contributions along x
    vec4 n_x = mix(vec4(n000, n001, n010, n011),
            vec4(n100, n101, n110, n111), fade(pf.x));
 
    // Blend contributions along y
    vec2 n_xy = mix(n_x.xy, n_x.zw, fade(pf.y));
 
    // Blend contributions along z
    float n_xyz = mix(n_xy.x, n_xy.y, fade(pf.z));
 
    // We're done, return the final noise value.
    return n_xyz;
}

// forward declaration
float rayIntersectScene( Ray ray, out Material out_mat, out Intersection out_intersect );

// Plane
// this function can be used for plane, triangle, and box
float findIntersectionWithPlane( Ray ray, vec3 norm, float dist, out Intersection intersect ) {
    float a   = dot( ray.direction, norm );
    float b   = dot( ray.origin, norm ) - dist;
    
    if ( a < 0.0 && a > 0.0 ) return INFINITY;
    
    float len = -b/a;
    if ( len < EPS ) return INFINITY;
    intersect.position = rayGetOffset( ray, len );
    intersect.normal   = norm;
    return len;
}

// Triangle
float findIntersectionWithTriangle(Ray ray, vec3 t1, vec3 t2, vec3 t3, out Intersection intersect) {
    // ask jeff what the distance in the plane intersection represents 
    vec3 v      = ray.direction;
    vec3 p0     = ray.origin; 
    vec3 n      = normalize(cross(t2 - t1, t3 - t2));
    float total = triangleArea(t1, t2, t3); 
    float dist  = dot(t1, n);
    float distance   = findIntersectionWithPlane(ray, n, dist, intersect);
    vec3 p           = intersect.position;
    float alpha      = triangleArea(t1, t2, p)/total;
    float beta       = triangleArea(t1, t3, p)/total;
    float gamma      = triangleArea(t2, t3, p)/total;
    if (dot(cross(t2 - t1, t3 - t1), n) < 0.0) return INFINITY;
    if (0.0 - EPS <= alpha + beta + gamma && alpha + beta + gamma <= 1.0 + EPS) return distance;
    return INFINITY;
     // currently reports no intersection
}

// Sphere
float findIntersectionWithSphere( Ray ray, vec3 center, float radius, out Intersection intersect ) {   
    vec3 p0            =   ray.origin; 
    vec3 l             =   center - p0; 
    vec3 v             =   ray.direction; 
    float tca          =   dot(l, v);
    if (tca < EPS) return INFINITY; 
    float d2           =   dot(l, l) - pow(tca, 2.0);
    float r2           =   pow(radius, 2.0); 
    if (d2 >= r2)  return INFINITY; 
    float thc          =   sqrt(r2 - d2);
    float t;
    if (tca - thc > EPS)      t =  tca - thc; 
    else if (tca + thc > EPS) t =  tca + thc; 
    else                      t =  INFINITY;
    vec3  p            =   p0 + t*v; 
    vec3  n            =   (p - center)/length(p - center); 
    intersect.position =   p;
    intersect.normal   =   normalize(n); 
    return t;
}

// Box
float findIntersectionWithBox(Ray ray, vec3 pmin, vec3 pmax, out Intersection out_intersect) {
    // pmin and pmax represent two bounding points of the box
    vec3 minx = vec3(pmin.x, 0.0, 0.0); 
    vec3 miny = vec3(0.0, pmin.y, 0.0);
    vec3 minz = vec3(0.0, 0.0, pmin.z);
    vec3 maxx = vec3(pmax.x, 0.0, 0.0);
    vec3 maxy = vec3(0.0, pmax.y, 0.0);
    vec3 maxz = vec3(0.0, 0.0, pmax.z);
    vec3 n1   = cross(minx, miny);
    vec3 n2   = cross(minx, minz);
    vec3 n3   = cross(miny, minz);
    vec3 n4   = cross(maxx, maxy);
    vec3 n5   = cross(maxx, maxz);
    vec3 n6   = cross(maxy, maxz);
    Intersection champion;
    Intersection current; 
    float dist          = dot(pmin, n1);
    float currDistance  = findIntersectionWithPlane(ray, n1, dist, current); 
    float champDistance = INFINITY;
    if (current.position.x >= pmin.x && current.position.x <= pmax.x &&
     current.position.y >= pmin.y && current.position.y <= pmax.y) {
        champion      = current;
        champDistance = currDistance;
    }
    dist          = dot(pmin, n2);
    currDistance  = findIntersectionWithPlane(ray, n2, dist, current); 
    if (current.position.x >= pmin.x && current.position.x <= pmax.x &&
     current.position.z >= pmin.z && current.position.z <= pmax.z && 
     currDistance < champDistance) {
        champion      = current;
        champDistance = currDistance;
    }
    dist          = dot(pmin, n3);
    currDistance  = findIntersectionWithPlane(ray, n3, dist, current); 
    if (current.position.y >= pmin.y && current.position.y <= pmax.y &&
     current.position.z >= pmin.z && current.position.z <= pmax.z && 
     currDistance < champDistance) {
        champion      = current;
        champDistance = currDistance;
    }
    dist          = dot(pmax, n4);
    currDistance  = findIntersectionWithPlane(ray, n4, dist, current); 
    if (current.position.x >= pmin.x && current.position.x <= pmax.x &&
     current.position.y >= pmin.y && current.position.y <= pmax.y && 
     currDistance < champDistance) {
        champion      = current;
        champDistance = currDistance;
    }
    dist          = dot(pmax, n5);
    currDistance  = findIntersectionWithPlane(ray, n5, dist, current); 
    if (current.position.x >= pmin.x && current.position.x <= pmax.x &&
     current.position.z >= pmin.z && current.position.z <= pmax.z && 
     currDistance < champDistance) {
        champion      = current;
        champDistance = currDistance;
    }
    dist          = dot(pmax, n6);
    currDistance  = findIntersectionWithPlane(ray, n6, dist, current); 
    if (current.position.y >= pmin.y && current.position.y <= pmax.y &&
     current.position.z >= pmin.z && current.position.z <= pmax.z && 
     currDistance < champDistance) {
        champion      = current;
        champDistance = currDistance;
    }
    out_intersect.position = champion.position;
    out_intersect.normal   = normalize(champion.normal);
    return champDistance;
}  

// Cylinder

float getIntersectOpenCylinder( Ray ray, vec3 center, vec3 axis, float len, float rad, out Intersection intersect ) {
    // What is the main difference between sphere and cylinder
    vec3 p      =   ray.origin ; 
    vec3 v      =   ray.direction;
    vec3 va     =   axis;
    vec3 pa     =   center; 
    vec3 deltap =   p - pa; 
    float vva   =   dot(v, va);  
    vec3  ua    =   v - vva*va;
    float a     =   dot(ua, ua);
    float dpva  =   dot(deltap, va); 
    vec3  uc    =   deltap - dpva*va; 
    float b     =   2.0*dot(ua, uc);
    float c     =   dot(uc, uc) - pow(rad, 2.0);
    // check both possibility that negative distances are allowed and possibility they are not allowed 
    float b24ac =   b*b - 4.0*a*c;
    if (b24ac < -1.0*EPS) return INFINITY;
    float t0    = (-1.0*b + sqrt(b24ac))/(2.0*a);
    float t1    = (-1.0*b - sqrt(b24ac))/(2.0*a);
    float t; 

    if (t0 >= -1.0*EPS && t1 >= -1.0*EPS) {
        t = min(t0, t1);
    } else if (t0 < EPS) {
        t = t1; 
    } else if (t1 < EPS) {
        t = t0; 
    } else {
        return INFINITY; 
    }
    
    vec3 pos = p + v*t;
    if (pos.y - center.y >= len) {
        return INFINITY;
    }
    intersect.position = pos;
    intersect.normal   = normalize((pos - center)/length(pos - center));
    return t;
     // currently reports no intersection
}

float getIntersectDisc( Ray ray, vec3 center, vec3 norm, float rad, out Intersection intersect ) {
    vec3 p0  = ray.origin; 
    vec3 v   = ray.direction;
    vec3 cent =  center;
    float dist = dot(cent, norm);
    float d = findIntersectionWithPlane(ray, norm, dist, intersect);
    if (length(intersect.position - center) < rad) {
        return d; 
    }
    return INFINITY; // currently reports no intersection
}


float findIntersectionWithCylinder( Ray ray, vec3 center, vec3 apex, float radius, out Intersection out_intersect ) {
    vec3 axis = apex - center;
    float len = length( axis );
    axis = normalize( axis );

    Intersection intersect;
    float best_dist = INFINITY;
    float dist;

    // -- infinite cylinder
    dist = getIntersectOpenCylinder( ray, center, axis, len, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    // -- two caps
    dist = getIntersectDisc( ray, center, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );
    dist = getIntersectDisc( ray,   apex, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );
    return best_dist;
}
    
// Cone
float getIntersectOpenCone( Ray ray, vec3 apex, vec3 axis, float len, float radius, out Intersection intersect ) {
    // What is the main difference between sphere and cylinder
    // vec3 p      =   ray.origin; 
    // vec3 v      =   ray.direction;     
    // float a     =   pow(v.x, 2.0) + pow(v.z, 2.0) - pow(v.y, 2.0);
    // float b     =   2.0*v.x*p.x + 2.0*v.z*p.z - 2.0*v.y*p.y;
    // float c     =   pow(p.x, 2.0) + pow(p.z, 2.0) - pow(v.y,2.0);
    // // check both possibility that negative distances are allowed and possibility they are not allowed 
    // float b24ac =   b*b - 4.0*a*c;
    // if (b24ac < 0.0) return INFINITY;
    // float t0    = (-1.0*b + sqrt(b24ac))/(2.0*a);
    // float t1    = (-1.0*b - sqrt(b24ac))/(2.0*a);
    // if (t0 > t1) {
    //     float tmp = t0; 
    //     t0 = t1; 
    //     t1 = tmp;
    // }
    // float y0    = p.y + t0 * v.y;
    // float y1    = p.y + t1 * v.y;
    // if ((y0 > len && y1 > len) || (y0 < 0.0 && y1 < 0.0)) return INFINITY;
    // if (y0 >= -1.0*len && y0 <= len) {
    //    if (t0 < 0.0)  return INFINITY;
    //    intersect.position = apex + axis*t0;
    //    intersect.normal   = normalize(vec3(intersect.position.x, 0.0, intersect.position.z));
    //    return t0;
    // }
    return INFINITY;
     // currently reports no intersection
}

float findIntersectionWithCone( Ray ray, vec3 center, vec3 apex, float radius, out Intersection out_intersect ) {
    vec3 axis   = center - apex;
    float len   = length( axis );
    axis = normalize( axis );
        
    // -- infinite cone
    Intersection intersect;
    float best_dist = INFINITY;
    float dist;

    // -- infinite cone
    dist = getIntersectOpenCone( ray, apex, axis, len, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    // -- caps
    dist = getIntersectDisc( ray, center, axis, radius, intersect );
    chooseCloserIntersection( dist, best_dist, intersect, out_intersect );

    return best_dist;
}

#define MAX_RECURSION 8

vec3 calculateSpecialDiffuseColor( Material mat, vec3 posIntersection, vec3 normalVector ) {
    if ( mat.special == CHECKERBOARD ) {
        // find vector perpendicular to normal
        vec3 trans      = normalize(posIntersection);
        float value     = floor(posIntersection.x/8.0) + floor(posIntersection.z/8.0) + floor(posIntersection.y/8.0); 
        if (mod(value, 2.0) == 0.0) {
            return mat.color;
        } 
        return mat.color * 0.5;
    
        // do something here for checkerboard
        // ----------- Our reference solution uses 21 lines of code.
    } 
    else if ( mat.special == MYSPECIAL ) {
        // do something here for myspecial
        return mat.color*pnoise3D(posIntersection);
    }

    return mat.color; // special materials not implemented. just return material color.
}

vec3 calculateDiffuseColor(Material mat, vec3 posIntersection, vec3 normalVector) {
    // Special colors
    if ( mat.special != NONE ) {
        return calculateSpecialDiffuseColor( mat, posIntersection, normalVector ); 
    }
    return vec3( mat.color );
}

// check if position pos in in shadow with respect to a particular light.
// lightVec is the vector from that position to that light
bool pointInShadow( vec3 pos, vec3 lightVec ) {
    Material mat;
    Intersection intersect;
    Ray ray = Ray(pos, normalize(lightVec));
    float dist = rayIntersectScene(ray, mat, intersect);
    if (dist >= 0.0 && length(lightVec) > dist) return true;
    return false;
}

float pointShadowRatio(vec3 pos, vec3 lightVec) {
    float count = 0.0;  
    for (int i = 0; i < 100; i++) {
        float x1 = 1.0;
        float x2 = 1.0;
        for (int j = 0; j < 30; j++) {
            x1 = rand(vec2(-1.0, 1.0));
            x2 = rand(vec2(-1.0, 1.0));
            if (x1*x1 + x2*x2 < 1.0) break; 
        }
        float x  = 2.0*x1*sqrt(1.0 - x1*x1 - x2*x2); 
        float y  = 2.0*x2*sqrt(1.0 - x1*x1 - x2*x2); 
        float z  = 1.0 - 2.0*(x1*x1 + x2*x2);
        vec3 newLightVec =vec3(x, y, z) + lightVec;
        if (!(pointInShadow(pos,  newLightVec))) {
            count += 1.0; 
        }
    }
    return count/float(100);
}
vec3 getLightContribution( Light light, Material mat, vec3 posIntersection, vec3 normalVector, vec3 eyeVector, bool phongOnly, vec3 diffuseColor ) {

    vec3 lightVector = light.position - posIntersection;
    
    // if ( pointInShadow( posIntersection, lightVector ) ) {
    //     return vec3( 0.0, 0.0, 0.0 );
    // }

    if ( mat.materialType == PHONGMATERIAL || mat.materialType == LAMBERTMATERIAL ) {
        vec3 contribution = vec3( 0.0, 0.0, 0.0 );

        // get light attenuation
        float dist = length( lightVector );
        float attenuation = light.attenuate * dist * dist;

        float diffuseIntensity = max( 0.0, dot( normalVector, lightVector ) ) * light.intensity;
        
        // glass and mirror objects have specular highlights but no diffuse lighting
        if ( !phongOnly ) {
            contribution += diffuseColor * diffuseIntensity * light.color / attenuation;
        }
        
        if ( mat.materialType == PHONGMATERIAL ) {
            // vec3 n = normalize(normalVector);
            // vec3 l = normalize(lightVector); 
            // vec3 r = normalize((2.0 * n * dot(n, l)) - l);
            // vec3 v = normalize(eyeVector);
            // float ks    =  mat.specular;
            // float alpha =  mat.shininess;
            // vec3 is     = light.color;
            // vec3 specular  = ks * pow(dot(v, r), alpha)*is;
            // vec3 phongTerm = specular + diffuseIntensity; // not implemented yet, so just add black   

            contribution += vec3( 0.0, 0.0, 0.0 );;
        }

        return contribution *  pointShadowRatio(posIntersection, lightVector);
    }
    else {
        return diffuseColor *  pointShadowRatio(posIntersection, lightVector);
    }

}

vec3 calculateColor( Material mat, vec3 posIntersection, vec3 normalVector, vec3 eyeVector, bool phongOnly ) {
	vec3 diffuseColor = calculateDiffuseColor( mat, posIntersection, normalVector );

	vec3 outputColor = vec3( 0.0, 0.0, 0.0 ); // color defaults to black when there are no lights
	
    for ( int i=0; i<MAX_LIGHTS; i++ ) {

        if( i>=numLights ) break; // because GLSL will not allow looping to numLights
		
        outputColor += getLightContribution( lights[i], mat, posIntersection, normalVector, eyeVector, phongOnly, diffuseColor );
	}
	
	return outputColor;
}

// find reflection or refraction direction ( depending on material type )
vec3 calcReflectionVector( Material material, vec3 direction, vec3 normalVector, bool isInsideObj ) {
    if( material.materialReflectType == MIRRORREFLECT ) {
        return reflect( direction, normalVector );
    }
    // the material is not mirror, so it's glass.
    // compute the refraction direction...
    // the eta below is eta_i/eta_r
    float eta       = ( isInsideObj ) ? 1.0/material.refractionRatio : material.refractionRatio;
    vec3 refractedV = refract(normalize(direction), normalVector, eta);
    float cos0      = dot(direction, normalVector)/(length(direction)*length(normalVector));
    float angle     = acos(cos0); 
    float critAngle = asin(eta);
    if (angle > critAngle) {
        return refractedV;
    } 
    return refractedV;
}

vec3 traceRay( Ray ray ) {
    Material hitMaterial;
    Intersection intersect;

    vec3 resColor  = vec3( 0.0, 0.0, 0.0 );
    vec3 resWeight = vec3( 1.0, 1.0, 1.0 );
    
    bool isInsideObj = false;

    for ( int depth = 0; depth < MAX_RECURSION; depth++ ) {
        
        float hit_length = rayIntersectScene( ray, hitMaterial, intersect );
            
        if ( hit_length < EPS || hit_length >= INFINITY ) break;

        vec3 posIntersection = intersect.position;
        vec3 normalVector    = intersect.normal;

        vec3 eyeVector = normalize( ray.origin - posIntersection );           
        if ( dot( eyeVector, normalVector ) < 0.0 )
            { normalVector = -normalVector; isInsideObj = true; }
        else isInsideObj = false;

        bool reflective = ( hitMaterial.materialReflectType == MIRRORREFLECT || 
                            hitMaterial.materialReflectType == GLASSREFLECT );
		vec3 outputColor = calculateColor( hitMaterial, posIntersection, normalVector, eyeVector, reflective );

        float reflectivity = hitMaterial.reflectivity;

        // check to see if material is reflective ( or refractive )
        if ( !reflective || reflectivity < EPS ) {
            resColor += resWeight * outputColor;
            break;
        }
        
        // bounce the ray
        vec3 reflectionVector = calcReflectionVector( hitMaterial, ray.direction, normalVector, isInsideObj );
        ray.origin = posIntersection;
        ray.direction = normalize( reflectionVector );

        // add in the color of the bounced ray
        resColor += resWeight * outputColor;
        resWeight *= reflectivity;
    }

    return resColor;
}

void main( ) {
    float cameraFOV = 0.8;
    vec3 direction = vec3( v_position.x * cameraFOV * width/height, v_position.y * cameraFOV, 1.0 );

    Ray ray;
	ray.origin    = vec3( uMVMatrix * vec4( camera, 1.0 ) );
    ray.direction = normalize( vec3( uMVMatrix * vec4( direction, 0.0 ) ) );


    // trace the ray for this pixel
    vec3 res = traceRay( ray );
    // paint the resulting color into this pixel
    gl_FragColor = vec4( res.x, res.y, res.z, 1.0 );
}

