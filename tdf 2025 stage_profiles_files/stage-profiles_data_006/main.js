/*
 * smart-designs v2.1
 * 22-8-2024
*/

"use strict";
var Premium = Premium || {};

Premium.creative = {
    init: function() {
        /* START OF CUSTOM JS */

        Premium.video.switchOnScroll(undefined, 2, function scrollHandler(pos) {
            if (pos === "up") {
                gsap.to('.side-copy', .5, { opacity: 0, ease: 'power3.out' })
            } else if (pos === "down") {
                gsap.to('.side-copy', .5, { opacity: 1, ease: 'power3.out' })
            }
        });

        gsap.set(".jpt-kv-image", { xPercent: -50, yPercent: -50 });

        Premium.communicator.init(document.body.id !== "body_expanded" ? 4 : undefined);

        Premium.animation.delay = document.body.id === "body_top" ? 1 : .5;
        var entranceAnimations = Premium.animation.getEntranceAnimation();
        Premium.communicator.api.receiveMessage(function(mess) {
            if (mess.action === "start-entrance-animation") {
                entranceAnimations.play();
            }
        });
        Premium.communicator.api.sendMessage({ action: "start-entrance-animation" });

        switch (document.body.id) {
            case "body_top":

                Premium.template.SetWidthsBaseWidth = 1480;
                Premium.template.SetWidthsMaxWidthRatio = 2.3;

                Premium.creative.createVideoPlayerAppended = function(video) {
                    Premium.video.sync(video, 1);
                    Premium.video.switchOnScroll(video.parentElement, 2);
                    PremiumJpControls.callOnClickFullScreen(function() {
                        video.muted = false;
                        Premium.expand.expand("expanded_video.html", "width:100%;height:100%;")
                    });
                    var prom = video.play();
                    if (prom) {
                        prom.catch(function() {})
                    }
                }

                break;

            case "body_left":

            case "body_right":

                Premium.template.SetWidthsBaseWidth = 1080;
                Premium.template.SetWidthsMaxWidthRatio = 0.5;

                var visualOnScroll = document.querySelector(".jpt-visual-onscroll");
                var insertWhenScrolled = document.querySelector(".jpt-insert-when-scrolled");

                Premium.creative.createVideoPlayerAppended = function(video, container) {
                    video.volume = 0;
                    PremiumJpControls.callOnClickFullScreen(function() {
                        video.muted = false;
                        Premium.expand.expand("expanded_video.html", "width:100%;height:100%;")
                    });
                    Premium.video.sync(video, 1);

                    Premium.video.switchOnScroll(document.querySelector(".jpt-video-container"), 3);
                    var prom = video.play();
                    if (prom) {
                        prom.catch(function() {})
                    }
                }

                gsap.set(".jpt-insert-when-scrolled", { marginBottom: Premium.template.getScrolledStateFlexMargin(), autoAlpha: 0 });

                Premium.video.switchOnScroll(undefined, 3, function(pos) {
                    if (visualOnScroll) {
                        if (pos === "down") {
                            gsap.timeline()
                                .to(".jpt-visual-onload", { duration: .4, autoAlpha: 0 }, 0)
                                .to(".jpt-visual-onscroll", { duration: .4, autoAlpha: 1, pointerEvents: "auto", onUpdate: PremiumJpControls.resizeAll }, 0)
                        } else {
                            gsap.timeline()
                                .to(".jpt-visual-onscroll", { duration: .4, autoAlpha: 0, pointerEvents: "none" }, 0)
                                .to(".jpt-visual-onload", { duration: .4, autoAlpha: 1 }, 0)
                        }
                    }
                    if (insertWhenScrolled) {
                        if (pos === "down") {
                            gsap.to(".jpt-insert-when-scrolled", .2, { marginBottom: "0", autoAlpha: 1 });
                        } else {
                            gsap.timeline().to(".jpt-insert-when-scrolled", .2, { autoAlpha: 0 })
                                .to(".jpt-insert-when-scrolled", .2, { marginBottom: Premium.template.getScrolledStateFlexMargin() }, "-=.17");
                        }
                    }
                });

                if (document.body.id === "body_left") {
                    // left panel-specific code here

                    // Pseudo Panels
                    // const sendResize = ()=>{
                    //     Premium.communicator.api.sendMessage({action:"resize"});
                    // }
                    // window.addEventListener("resize", sendResize);
                    // window.addEventListener("load", sendResize)

                } else if (document.body.id === "body_right") {
                    // right panel-specific code here
                }

                break;


            case "body_back":
                // back panel code here

                // Pseudo Panels
                // Premium.communicator.api.receiveMessage(function(mess) {
                //     if (mess.action === "resize") {
                //         var siteWidth = Premium.product.creativeMainEl().clientWidth;
                //         var oneSide = (window.innerWidth - siteWidth) / 2;
                //         gsap.set(".pseudo-top", { width: siteWidth, height: Premium.product.creativeMainEl().clientHeight, left: oneSide });
                //         gsap.set(".pseudo-left, .pseudo-right", { width: oneSide });
                //     }
                // })
                // Premium.communicator.api.sendMessage({action:"resize"});

                break;

            case "body_expanded":

                if (document.querySelector(".jpt-section-video")) {
                    Premium.creative.createVideoPlayerAppended = function(video) {
                        video.volume = 0;
                        Premium.video.sync(video, undefined, Premium.video.SyncType_Get);
                        video.addEventListener("playing", function() {
                            // setTimeout(function(){
                            document.body.style.display = "block";
                            document.body.style.opacity = 1;
                            // }, 200)  
                        }, { once: true })
                        var prom = video.play();
                        if (prom) {
                            prom.catch(function() {})
                        }
                        // Safari fix
                        document.body.addEventListener("click", function(e) {
                            if (e.target.className && (e.target.className.indexOf("jp-controls-play") > -1 || e.target.className.indexOf("jp-controls-bigplay") > -1)) {
                                if (!video.paused) {
                                    setTimeout(function() {
                                        video.play();
                                    }, 800)
                                }
                            }
                        });
                    }
                } else {
                    document.body.style.display = "block";
                    document.body.style.opacity = 1;
                }

                break;
        }

        Premium.template.setImageWidths();

        /* END OF CUSTOM JS */
    },
    loaded: function() {
        document.body.style.opacity = 1;
    }
}