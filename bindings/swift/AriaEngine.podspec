Pod::Spec.new do |s|
  s.name             = 'AriaEngine'
  s.version          = '0.1.0'
  s.summary          = 'Aria Engine Swift binding'
  s.homepage         = 'https://github.com/ariacompute/engine'
  s.license          = { :type => 'MIT' }
  s.author           = { 'Aria Compute' => 'hello@ariacompute.com' }
  s.source           = { :git => 'https://github.com/ariacompute/engine.git', :tag => s.version.to_s }
  s.swift_version    = '5.9'
  s.ios.deployment_target = '15.0'
  s.osx.deployment_target = '12.0'
  s.source_files     = 'Sources/AriaEngine/**/*.swift'
  s.vendored_frameworks = 'AriaFFI.xcframework' # attached on Release
end
