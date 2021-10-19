/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// tslint:disable:no-new-decorators
import {action, observable} from 'mobx';

import {LayoutSettings, LitComponentLayout, LitComponentSpecifier, LitModuleClass, ModelInfoMap, Spec} from '../lib/types';

import {LitService} from './lit_service';
import {ModulesObservedByUrlService, UrlConfiguration} from './url_service';

/**
 * A layout is defined by a set of main components that are always visible,
 * (designated in the object by the "main" key)
 * and a set of tabs that each contain a group other components.
 *
 * LitRenderConfig is a mapping of tab names arrays of component lists.
 * Each component list includes multiple instances of the same component,
 * duplicated for each model. This basically corresponds to the grid of
 * components that will be rendered.
 */
export interface LitRenderConfig {
  [name: string]: RenderConfig[][];
}

/**
 * An interface describing how to render a LIT module, specifying which module
 * to render and whether it renders on a per-model basis.
 */
export interface RenderConfig {
  key: string;
  moduleType: LitModuleClass;
  modelName?: string;
  selectionServiceIndex?: number;
}

type RenderModulesCallback = () => void;

/**
 * Look up any module names given as strings, and return the
 * constructor object.
 */
export function getModuleConstructor(moduleType: LitComponentSpecifier):
    LitModuleClass {
  if (typeof moduleType === 'string') {
    const moduleClass = window.customElements.get(moduleType);
    if (moduleClass === undefined) {
      throw (new Error(
          `Malformed layout; unable to find element '${moduleType}'`));
    }
    return moduleClass as unknown as LitModuleClass;
  }
  return moduleType;
}

/**
 * Singleton service responsible for maintaining which modules to render.
 */
export class ModulesService extends LitService implements
    ModulesObservedByUrlService {
  @observable
  declaredLayout: LitComponentLayout = {'components': {}, 'layoutSettings': {}};
  @observable selectedTab: string = '';
  private renderLayout: LitRenderConfig = {};
  private renderModulesCallback: RenderModulesCallback = () => {};

  /**
   * We need to make the rendering of modules an explicit, callback-driven
   * update because of difficulty ensuring the template results of the
   * lit-modules don't trigger detach/reattach behavior of the module
   * components.
   */
  setRenderModulesCallback(callback: RenderModulesCallback) {
    this.renderModulesCallback = callback;
  }

  /**
   * Explicitly calls the render modules callback, which is set to rerender
   * the lit-modules component.
   */
  renderModules() {
    this.renderModulesCallback();
  }

  /**
   * In app initialization, we need to set the declared layout and compute the
   * visible render layout based on the app config
   */
  initializeLayout(
      layout: LitComponentLayout, currentModelSpecs: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean) {
    this.declaredLayout = layout;
    this.declaredLayout.layoutSettings = layout.layoutSettings || {};
    this.updateRenderLayout(currentModelSpecs, datasetSpec, compareExamples);
  }
  @observable hiddenModuleKeys = new Set<string>();
  allModuleKeys = new Set<string>();

  @action
  clearLayout() {
    this.updateRenderLayout({}, {}, /* compareExamples */ false);
    this.renderModules();
  }

  /**
   * Recompute layout without clearing modules.
   * Use this for comparison mode to provide a faster refresh, and avoid
   * clearing module-level state such as datatable filters.
   */
  @action
  quickUpdateLayout(
      currentModelSpecs: ModelInfoMap, datasetSpec: Spec,
      compareExamples: boolean) {
    // Recompute layout
    this.updateRenderLayout(currentModelSpecs, datasetSpec, compareExamples);
    this.renderModules();
  }

  setHiddenModules(keys: Set<string>|string[]) {
    // Ensure we copy to a new set
    const nextHiddenModuleKeys = new Set<string>(keys);
    this.hiddenModuleKeys = nextHiddenModuleKeys;
  }

  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.setHiddenModules(urlConfiguration.hiddenModules);
    this.selectedTab = urlConfiguration.selectedTab ?? '';
  }

  isModuleGroupHidden(config: RenderConfig) {
    return this.hiddenModuleKeys.has(config.key);
  }

  toggleHiddenModule(config: RenderConfig, isHidden: boolean) {
    if (isHidden) {
      this.hiddenModuleKeys.add(config.key);
    } else {
      this.hiddenModuleKeys.delete(config.key);
    }
  }

  getRenderLayout() {
    return this.renderLayout;
  }

  getSetting(settingName: keyof LayoutSettings) {
    return this.declaredLayout?.layoutSettings?.[settingName];
  }

  /**
   * Compute module configurations to render determining whether or not a module
   * is visible for the selected models and user settings, and whether to render
   * copies of a module per model based on the module behavior.
   */
  updateRenderLayout(
      currentModelSpecs: ModelInfoMap, datasetSpec: Spec,
      compareExamples: boolean) {
    const renderLayout: LitRenderConfig = {};
    const allModuleKeys = new Set<string>();

    const componentGroupNames = Object.keys(this.declaredLayout.components);
    for (const groupName of componentGroupNames) {
      const components = this.declaredLayout.components[groupName];
      // Look up classes for this group, if anything is given as a string.
      const componentClasses = components.map(getModuleConstructor);
      // First, map all of the modules to render configs, filtering out those
      // that are not visible.
      const configs = this.getRenderConfigs(
          componentClasses, currentModelSpecs, datasetSpec, compareExamples,
          groupName);
      for (const configGroup of configs) {
        for (const config of configGroup) {
          allModuleKeys.add(config.key);

          if (config.moduleType.collapseByDefault) {
            this.hiddenModuleKeys.add(config.key);
          }
        }
      }
      if (configs.length !== 0) {
        renderLayout[groupName] = configs;
      }
    }
    this.allModuleKeys = allModuleKeys;

    // Clean up any extraneous hidden module keys that are not part of the
    // allModuleKeys set
    for (const key of [...this.hiddenModuleKeys]) {
      if (!this.allModuleKeys.has(key)) {
        this.hiddenModuleKeys.delete(key);
      }
    }
    this.renderLayout = renderLayout;
  }

  /**
   * Generates module renderConfig object or objects for a given LIT module
   * depending on the model specs. Since some modules can render one copy per
   * model, this method specifies the configurations for those multiple modules
   * to render.
   */
  private getRenderConfigs(
      modules: LitModuleClass[], currentModelSpecs: ModelInfoMap,
      datasetSpec: Spec, compareExamples: boolean,
      tab: string): RenderConfig[][] {
    const renderConfigs: RenderConfig[][] = [];
    // Iterate over all modules to generate render config objects, expanding
    // modules that display one per model.
    for (const moduleType of modules) {
      if (!moduleType.shouldDisplayModule(currentModelSpecs, datasetSpec)) {
        continue;
      }

      const configs: RenderConfig[] = [];

      // Is compare examples mode on and does this module require duplication?
      const compare =
          compareExamples && moduleType.duplicateForExampleComparison;
      const key = `${tab}_${moduleType.title}`;

      // model = undefined means the resulting module(s) will not be keyed to
      // a specific model; they can access the list of active models via
      // this.appState.currentModels.
      let selectedModels: Array<string|undefined> = [undefined];
      if (moduleType.duplicateForModelComparison) {
        selectedModels = Object.keys(currentModelSpecs);
      }
      for (const modelName of selectedModels) {
        if (compare) {
          // The 'reference' selection service is index 1, but we want this to
          // render on top/left, so create this config first.
          configs.push(this.makeRenderConfig(key, moduleType, modelName, 1));
          configs.push(this.makeRenderConfig(key, moduleType, modelName, 0));
        } else {
          configs.push(this.makeRenderConfig(
              key, moduleType, modelName, compareExamples ? 0 : undefined));
        }
      }
      renderConfigs.push(configs);
    }

    return renderConfigs;
  }

  private makeRenderConfig(
      key: string, moduleType: LitModuleClass, modelName?: string,
      selectionServiceIndex?: number): RenderConfig {
    return {
      key,
      moduleType,
      modelName,
      selectionServiceIndex,
    };
  }
}
