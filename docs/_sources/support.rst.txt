.. _support:

Support and contributing
************************

.. _ways-to-contact-us:

Ways to contact us
------------------

If you find an apparent issue with cf-plot or have a question or feature
request etc., the *preferred* method of contact is via the
`GitHub Issue Tracker for the cf-plot repository <https://github.com/NCAS-CMS/cf-plot/issues>`_.
You will need a GitHub account to do this, but it is free to
`sign up <https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github>`_.

If you don't want to use the Issue Tracker then you can also either:

* send an email to Sadie Bartholomew at `sadie.bartholomew@ncas.ac.uk` (or
  if based on the University of Reading campus you may find her in-office
  HP113, noting she often works from home);
* post on the `NCAS-CMS Helpdesk <https://cms-helpdesk.ncas.ac.uk/>`_.

If you are reporting an issue, please follow the guidelines below so that
we have enough information to attempt to recreate the problem.


.. raw:: html

   <!-- omit in toc -->


.. _contrib:

Contributing to cf-plot
-----------------------

We have a dedicated
`contributing document <https://github.com/NCAS-CMS/cf-plot/blob/main/.github/CONTRIBUTING.md>`_
which outlines guidance for contributing. This information has also been
included below for ease of reference.

.. Below should be kept identical to the CONTRIBUTING.md except in .rst
   format. It is simplest to write into CONTRIBUTING.md, use a markdown
   to RST converter and then copy and paste that in. So, don't edit this
   section directly, edit there are use Pandoc to convert i.e:
   pandoc -s CONTRIBUTING.md -o CONTRIBUTING.rst ! Note will need to adapt
   heading levels to get sub-sectioning right.


First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the `Table of
Contents <#table-of-contents>`__ for different ways to help and details
about how this project handles them. Please make sure to read the
relevant section before making your contribution. It will make it a lot
easier for us maintainers and smooth out the experience for all
involved. The community looks forward to your contributions.

   And if you like the project, but just don’t have time to contribute,
   that’s fine. There are other easy ways to support the project and
   show your appreciation, which we would also be very happy about: -
   Star the project - Tweet about it - Refer this project in your
   project’s readme - Mention the project at local meetups and tell your
   friends/colleagues

.. raw:: html

   <!-- omit in toc -->

Table of Contents
~~~~~~~~~~~~~~~~~

-  `Code of Conduct <#code-of-conduct>`__
-  `I Have a Question <#i-have-a-question>`__

   -  `I Want To Contribute <#i-want-to-contribute>`__
   -  `Reporting Bugs <#reporting-bugs>`__
   -  `Suggesting Enhancements <#suggesting-enhancements>`__
   -  `Your First Code Contribution <#your-first-code-contribution>`__
   -  `Improving The Documentation <#improving-the-documentation>`__

-  `Styleguides <#styleguides>`__

   -  `Commit Messages <#commit-messages>`__

-  `Join The Project Team <#join-the-project-team>`__

Code of Conduct
~~~~~~~~~~~~~~~

This project and everyone participating in it is governed by the
`cf-plot Code of
Conduct <https://github.com/NCAS-CMS/cf-plot/blob/main/CODE_OF_CONDUCT.md>`__.
By participating, you are expected to uphold this code. Please report
unacceptable behavior to sadie.bartholomew@ncas.ac.uk.

I Have a Question
~~~~~~~~~~~~~~~~~

   If you want to ask a question, we assume that you have read the
   available `Documentation <https://ncas-cms.github.io/cf-plot/>`__.

Before you ask a question, it is best to search for existing
`Issues <https://github.com/NCAS-CMS/cf-plot/issues>`__ that might help
you. In case you have found a suitable issue and still need
clarification, you can write your question in this issue.

If you then still feel the need to ask a question and need
clarification, we recommend the following, open an
`Issue <https://github.com/NCAS-CMS/cf-plot/issues/new>`__. Please use
the ‘general question’ template in this case, where further guidance can
be found as to how to structure your report and what to include. In
general, though: - Provide as much context as you can about what you’re
running into. - Provide project and platform versions (nodejs, npm,
etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

.. raw:: html

   <!--
   You might want to create a separate issue tag for questions and include it in this description. People should then tag their issues accordingly.

   Depending on how large the project is, you may want to outsource the questioning, e.g. to Stack Overflow or Gitter. You may add additional contact and information possibilities:
   - IRC
   - Slack
   - Gitter
   - Stack Overflow tag
   - Blog
   - FAQ
   - Roadmap
   - E-Mail List
   - Forum
   -->

I Want To Contribute
~~~~~~~~~~~~~~~~~~~~

   .. rubric:: Legal Notice
      :name: legal-notice

   When contributing to this project, you must agree that you have
   authored 100% of the content, that you have the necessary rights to
   the content and that the content you contribute may be provided under
   the project licence.

Reporting Bugs
^^^^^^^^^^^^^^

.. raw:: html

   <!-- omit in toc -->

Before Submitting a Bug Report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A good bug report shouldn’t leave others needing to chase you up for
more information. Therefore, we ask you to investigate carefully,
collect information and describe the issue in detail in your report.
Please complete the following steps in advance to help us fix any
potential bug as fast as possible.

-  Make sure that you are using the latest version.
-  Determine if your bug is really a bug and not an error on your side
   e.g. using incompatible environment components/versions (Make sure
   that you have read the
   `documentation <https://ncas-cms.github.io/cf-plot/>`__. If you are
   looking for support, you might want to check `this
   section <#i-have-a-question>`__).
-  To see if other users have experienced (and potentially already
   solved) the same issue you are having, check if there is not already
   a bug report existing for your bug or error in the `bug
   tracker <https://github.com/NCAS-CMS/cf-plot/issues?q=label%3Abug>`__.
-  Also make sure to search the internet (including Stack Overflow) to
   see if users outside of the GitHub community have discussed the
   issue.
-  Collect information about the bug:

   -  Stack trace (Traceback)
   -  OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
   -  Version of the interpreter, compiler, SDK, runtime environment,
      package manager, depending on what seems relevant.
   -  Possibly your input and the output
   -  Can you reliably reproduce the issue? And can you also reproduce
      it with older versions?

.. raw:: html

   <!-- omit in toc -->

How Do I Submit a Good Bug Report?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   You must never report security related issues, vulnerabilities or
   bugs including sensitive information to the issue tracker, or
   elsewhere in public. Instead sensitive bugs must be sent by email to
   sadie.bartholomew@ncas.ac.uk.

We use GitHub issues to track bugs and errors. If you run into an issue
with the project, open an
`Issue <https://github.com/NCAS-CMS/cf-plot/issues/new>`__. Please use
the ‘bug report’ template in this case, where further guidance can be
found as to how to structure your report and what to include. In
general, though:

-  Explain the behavior you would expect and the actual behavior.
-  Please provide as much context as possible and describe the
   *reproduction steps* that someone else can follow to recreate the
   issue on their own. This usually includes your code. For good bug
   reports you should isolate the problem and create a reduced test
   case.
-  Provide the information you collected in the previous section.

Once it’s filed:

-  The project team will respond to you as soon as they practicably can.

.. raw:: html

   <!-- You might want to create an issue template for bugs and errors that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

Suggesting Enhancements
^^^^^^^^^^^^^^^^^^^^^^^

This section guides you through submitting an enhancement suggestion for
cf-plot, **including completely new features and minor improvements to
existing functionality**. Following these guidelines will help
maintainers and the community to understand your suggestion and find
related suggestions.

.. raw:: html

   <!-- omit in toc -->

Before Submitting an Enhancement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-  Make sure that you are using the latest version.
-  Read the `documentation <https://ncas-cms.github.io/cf-plot/>`__
   carefully and find out if the functionality is already covered, maybe
   by an individual configuration.
-  Perform a `search <https://github.com/NCAS-CMS/cf-plot/issues>`__ to
   see if the enhancement has already been suggested. If it has, add a
   comment to the existing issue instead of opening a new one.
-  Find out whether your idea fits with the scope and aims of the
   project. It’s up to you to make a strong case to convince the
   project’s developers of the merits of this feature. Keep in mind that
   we want features that will be useful to the majority of our users and
   not just a small subset. If you’re just targeting a minority of
   users, consider writing an add-on/plugin library.

.. raw:: html

   <!-- omit in toc -->

How Do I Submit a Good Enhancement Suggestion?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Enhancement suggestions are tracked as `GitHub
issues <https://github.com/NCAS-CMS/cf-plot/issues>`__. Please use the
‘feature request’ template in this case, where further guidance can be
found as to how to structure your report and what to include. In
general, though:

-  Use a **clear and descriptive title** for the issue to identify the
   suggestion.
-  Provide a **step-by-step description of the suggested enhancement**
   in as many details as possible.
-  **Describe the current behavior** and **explain which behavior you
   expected to see instead** and why. At this point you can also tell
   which alternatives do not work for you.
-  You may want to **include screenshots or screen recordings** which
   help you demonstrate the steps or point out the part which the
   suggestion is related to. You can use
   `LICEcap <https://www.cockos.com/licecap/>`__ to record GIFs on macOS
   and Windows, and the built-in `screen recorder in
   GNOME <https://help.gnome.org/users/gnome-help/stable/screen-shot-record.html.en>`__
   or `SimpleScreenRecorder <https://github.com/MaartenBaert/ssr>`__ on
   Linux.
-  **Explain why this enhancement would be useful** to most cf-plot
   users. You may also want to point out the other projects that solved
   it better and which could serve as inspiration.

.. raw:: html

   <!-- You might want to create an issue template for enhancement suggestions that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

Other ways to contribute
^^^^^^^^^^^^^^^^^^^^^^^^

Besides bug reporting and enhancement suggestions, you can contribute in
other ways such as by improving the documentation through extending it
or updating it with new material, including writing some new code
recipe(s) for the `cf-python and cf-plot
recipes <https://ncas-cms.github.io/cf-python/recipes/index.html>`__
which are showcased in the cf-python documentation.

.. raw:: html

   <!-- omit in toc -->

Attribution
~~~~~~~~~~~

This guide is based on the
`contributing.md <https://contributing.md/generator>`__!
