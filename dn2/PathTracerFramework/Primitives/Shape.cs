﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Interpolation;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PathTracer
{
    public abstract class Primitive
    {
        public Transform ObjectToWorld { get; set; }
        public Transform WorldToObject => ObjectToWorld.Inverse();
        public abstract (double?, SurfaceInteraction) Intersect(Ray r);
        public abstract (SurfaceInteraction, double) Sample();

    }

    /// <summary>
    /// Base Shape class
    /// </summary>
    public abstract class Shape : Primitive
    {
        public BSDF BSDF { get; set; } = new BSDF();
        public abstract double Area();

        /// <summary>
        /// Estimates pdf of wi starting from point si
        /// </summary>
        /// <param name="si">point on surface that wi starts from</param>
        /// <param name="wi">wi</param>
        /// <returns>pdf of wi given this shape</returns>
        public virtual double Pdf(SurfaceInteraction si, Vector3 wi)
        {
            // Intersect sample ray with area light geometry
            Ray ray = si.SpawnRay(wi);
            (double? tHit, SurfaceInteraction isectLight) = Intersect(ray);

            if (!tHit.HasValue)
                return 0;

            // Convert area measure to solid angle measure
            double pdf = (si.Point - isectLight.Point).LengthSquared() / (Vector3.AbsDot(isectLight.Normal, -wi) * Area());
            if (double.IsInfinity(pdf))
                pdf = 0;
            return pdf;
        }

        /// <summary>
        /// Samples point on shape and returns vector to the given point
        /// </summary>
        /// <param name="si">Point the vector should start from</param>
        /// <returns>point on shape and its pdf</returns>
        public (SurfaceInteraction, double) Sample(SurfaceInteraction si)
        {
            (SurfaceInteraction intr, double pdf) = Sample();
            var wi = intr.Point - si.Point;
            if (wi.LengthSquared() < Renderer.Epsilon)
                pdf = 0;
            else
            {
                wi.Normalize();
                // Convert from area measure, as returned by the primitive Sample() call above, to solid angle measure.
                pdf *= (si.Point - intr.Point).LengthSquared() / Vector3.AbsDot(intr.Normal, -wi);
                if (double.IsInfinity(pdf))
                    pdf = 0;
            }
            return (intr, pdf);
        }
    }

}
