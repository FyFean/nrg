using System;
using MathNet.Numerics.Integration;

namespace PathTracer
{
    /// <summary>
    /// Sphere Shape template class - NOT implemented completely
    /// </summary>
    class Sphere : Shape
    {
        public double Radius { get; set; }
        public Sphere(double radius, Transform objectToWorld)
        {
            Radius = radius;
            ObjectToWorld = objectToWorld;
        }

        /// <summary>
        /// Ray-Sphere intersection - NOT implemented
        /// </summary>
        /// <param name="r">Ray</param>
        /// <returns>t or null if no hit, point on surface</returns>
        public override (double?, SurfaceInteraction) Intersect(Ray ray)
        {
            Ray r = WorldToObject.Apply(ray);

            // TODO: Compute quadratic sphere coefficients
            // Vector3 a = Sqr(di.x) + Sqr(di.y) + Sqr(di.z);
            double a = Vector3.Dot(r.d, r.d);
            // Interval b = 2 * (di.x * oi.x + di.y * oi.y + di.z * oi.z);
            double b = 2 * Vector3.Dot(r.d, r.o);
            // Interval c = Sqr(oi.x) + Sqr(oi.y) + Sqr(oi.z) - Sqr(Interval(radius));
            double c = Vector3.Dot(r.o, r.o) - Radius * Radius;

            // TODO: Initialize _double_ ray coordinate values
            double t0, t1;
            bool solv;

            // TODO: Solve quadratic equation for _t_ values
            (solv, t0, t1) = Utils.Quadratic(a, b, c);
            if (!solv) return (null, null);

            // TODO: Check quadric shape _t0_ and _t1_ for nearest intersection
            if (t0 > t1) (t0, t1) = (t1, t0);   // swap


            if (t0 < 0)
            {
                t0 = t1; // If t0 is negative, let's use t1 instead.
                if (t0 < 0) return (null, null); // Both t0 and t1 are negative.
            }


            // TODO: Compute sphere hit position and $\phi$
            Vector3 pHit = r.o + t0 * r.d;
            // if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * Radius;     // in case phit is at the center
            // Float phi = Math.atan2(pHit.y, pHit.x);
            // if (phi < 0) phi += 2 * Math.Pi;     // [0,2pi]

            // TODO: Return shape hit and surface interaction


            Vector3 n = pHit.Clone().Normalize();
            Vector3 dpdu = new Vector3(-pHit.y, pHit.x, 0);
            return (t0, ObjectToWorld.Apply(new SurfaceInteraction(pHit, n, -r.d, dpdu, this)));


            // A dummy return example
            // double dummyHit = 0.0;
            // Vector3 dummyVector = new Vector3(0, 0, 0);
            // SurfaceInteraction dummySurfaceInteraction = new SurfaceInteraction(dummyVector, dummyVector, dummyVector, dummyVector, this);
            // return (dummyHit, dummySurfaceInteraction);
        }

        /// <summary>
        /// Sample point on sphere in world
        /// </summary>
        /// <returns>point in world, pdf of point</returns>
        public override (SurfaceInteraction, double) Sample()
        {
            // TODO: Implement Sphere sampling
            Vector3 v = Samplers.UniformSampleSphere();

            // TODO: Return surface interaction and pdf
            var pObj = new Vector3(v.x * Radius, v.y * Radius, v.z * Radius);
            //var pObj = Samplers.UniformSampleSphere();

            var dpdu = new Vector3(-pObj.y, pObj.x, 0);
            //var n = ObjectToWorld.ApplyNormal(pObj); // Normalized vector since we're sampling on the surface of the sphere
            var n = pObj.Clone().Normalize();

            Debug.WriteLine("Sample sphere " + pObj.x + " " + pObj.y + " " + pObj.z);



            // Apply the object-to-world transformation to the surface interaction Vector3 point, Vector3 normal, Vector3 wo, Vector3 dpdu, Primitive obj
            SurfaceInteraction si = ObjectToWorld.Apply(new SurfaceInteraction(pObj, n, Vector3.ZeroVector, dpdu, this));
            double pdf = 1 / Area();

            return (si, pdf);



            // A dummy return example
            // double dummyPdf = 1.0;
            // Vector3 dummyVector = new Vector3(0, 0, 0);
            // SurfaceInteraction dummySurfaceInteraction = new SurfaceInteraction(dummyVector, dummyVector, dummyVector, dummyVector, this);
            // return (dummySurfaceInteraction, dummyPdf);
        }


        public override double Area() { return 4 * Math.PI * Radius * Radius; }

        /// <summary>
        /// Estimates pdf of wi starting from point si
        /// </summary>
        /// <param name="si">point on surface that wi starts from</param>
        /// <param name="wi">wi</param>
        /// <returns>pdf of wi given this shape</returns>
        public override double Pdf(SurfaceInteraction si, Vector3 wi)
        {
            throw new NotImplementedException();
        }

    }
}
