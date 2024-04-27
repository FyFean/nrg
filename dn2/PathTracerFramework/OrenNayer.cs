using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PathTracer
{
    /// <summary>
    /// Example BxDF implementation of a perfect OrenNayer surface
    /// </summary>
    public class OrenNayer : BxDF
    {
        private Spectrum kd;
        private double sigma;

        public OrenNayer(Spectrum r, double sig)
        {
            kd = r;
            sigma = sig;
        }

        /// <summary>
        /// OrenNayer f is kd/pi * A + B max(0, cos(φi − φo)) * sinα * tanβ)
        /// </summary>
        /// <param name="wo">output vector</param>
        /// <param name="wi">input vector</param>
        /// <returns></returns>
        public override Spectrum f(Vector3 wo, Vector3 wi)
        {
            if (!Utils.SameHemisphere(wo, wi))
                return Spectrum.ZeroSpectrum;

            double ss = sigma * sigma;
            double A = 1 - (ss / (2 * (ss + 0.33)));
            double B = 0.45 * (ss / (ss + 0.33));
            double theta_i = Math.Acos(wi.z / Math.Sqrt(wi.x * wi.x + wi.y * wi.y + wi.z * wi.z));
            double theta_o = Math.Acos(wo.z / Math.Sqrt(wo.x * wo.x + wo.y * wo.y + wo.z * wo.z));
            double phi_i = Math.Atan2(wi.y, wi.x);
            double phi_o = Math.Atan2(wo.y, wo.x);

            double alpha = Math.Max(theta_i, theta_o);
            double beta = Math.Min(theta_i, theta_o);

            //Spectrum f = Spectrum.Create(this.kd * Utils.PiInv * (A + B * Math.Max(0, Math.Cos(phi_i - phi_o)) * Math.Sin(alpha) * Math.Tan(beta)));
            return (this.kd * Utils.PiInv) * (A + B * Math.Max(0, Math.Cos(phi_i - phi_o)) * Math.Sin(alpha) * Math.Tan(beta));

        }
        /// <summary>
        /// Cosine weighted sampling of wi
        /// </summary>
        /// <param name="wo">wo in local</param>
        /// <returns>(f, wi, pdf)</returns>
        public override (Spectrum, Vector3, double) Sample_f(Vector3 wo)
        {
            var wi = Samplers.CosineSampleHemisphere();
            if (wo.z < 0)
                wi.z *= -1;
            double pdf = Pdf(wo, wi);
            return (f(wo, wi), wi, pdf);
        }

        /// <summary>
        /// returns pdf(wo,wi) as |cosTheta|/pi
        /// </summary>
        /// <param name="wo">output vector in local</param>
        /// <param name="wi">input vector in local</param>
        /// <returns></returns>
        public override double Pdf(Vector3 wo, Vector3 wi)
        {
            if (!Utils.SameHemisphere(wo, wi))
                return 0;

            return Math.Abs(wi.z) * Utils.PiInv; // wi.z == cosTheta
        }
    }
}
